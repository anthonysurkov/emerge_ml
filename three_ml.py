import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as pyg_softmax
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import glob
import gc
import os
from collections import OrderedDict, defaultdict
from torch.utils.data import Sampler

NUM_SEQS = 65536
NUM_EPOCHS = 100
NUM_SUBOPTS = 64
TARGET_ID = 'r270x_z'

PT_DATA_INFILE = f"data/training/{TARGET_ID}_subopts_{NUM_SEQS}_k{NUM_SUBOPTS}.pt"
MODEL_OUTFILE = f"data/models/{TARGET_ID}_{NUM_SEQS}_k{NUM_SUBOPTS}_e{NUM_EPOCHS}.pt"
PERFORMANCE_DATA_OUTFILE = f"data/performance/{TARGET_ID}_{NUM_SEQS}_k{NUM_SUBOPTS}_e{NUM_EPOCHS}.npz"

# Checkpoint file encodes all hyperparameters so a changed config won't
# accidentally resume from a stale save.
CHECKPOINT_FILE = f"data/models/{TARGET_ID}_{NUM_SEQS}_k{NUM_SUBOPTS}_e{NUM_EPOCHS}_ckpt.pt"

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

EXTRA_NODE_FEATS = 6
K_SUBSAMPLE_TRAIN = 64
K_FULL_EVAL = None

CHUNK_CACHE_SIZE = 3


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def save_checkpoint(epoch, model, optim, scheduler, train_losses, val_losses, path):
    """Save mid-training state so a crash can be resumed from the last epoch."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,                          # last *completed* epoch
            "model_state": model.state_dict(),
            "optim_state": optim.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        path,
    )


def load_checkpoint(path, model, optim, scheduler):
    """
    Load a checkpoint into model/optim/scheduler in-place.
    Returns (start_epoch, train_losses, val_losses).
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optim.load_state_dict(ckpt["optim_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    print(
        f"[checkpoint] Resumed from epoch {ckpt['epoch'] + 1} "
        f"(best val so far: {min(ckpt['val_losses']):.6f})"
    )
    return ckpt["epoch"] + 1, ckpt["train_losses"], ckpt["val_losses"]


# ── Dataset / sampler (unchanged) ─────────────────────────────────────────

class LRUChunkCache:
    def __init__(self, paths: list, maxsize: int):
        self._paths = paths
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def get(self, chunk_idx: int) -> dict:
        if chunk_idx in self._cache:
            self._cache.move_to_end(chunk_idx)
            return self._cache[chunk_idx]
        data = torch.load(self._paths[chunk_idx], map_location="cpu")
        if len(self._cache) >= self._maxsize:
            self._cache.popitem(last=False)
        self._cache[chunk_idx] = data
        return data

    @property
    def num_chunks(self):
        return len(self._paths)


class ChunkedViennaLandscapeDataset(Dataset):
    def __init__(self, chunk_dir: str, pattern: str = "*_chunk*.pt"):
        super().__init__()
        paths = sorted(glob.glob(f"{chunk_dir}/{pattern}"))
        if not paths:
            raise FileNotFoundError(
                f"No chunk files found in {chunk_dir!r} matching {pattern!r}"
            )

        self._offsets: list[tuple[int, int]] = []
        _x_shape = None
        for ci, p in enumerate(paths):
            tmp = torch.load(p, map_location="cpu")
            n_seqs = int(tmp["seq_ptr"].numel()) - 1
            for li in range(n_seqs):
                self._offsets.append((ci, li))
            if _x_shape is None:
                _x_shape = tmp["X"].shape
            del tmp
            gc.collect()

        self._x_shape = _x_shape
        self._cache = LRUChunkCache(paths, maxsize=CHUNK_CACHE_SIZE)
        self._mode = "train"
        self._K_train = K_SUBSAMPLE_TRAIN
        self._K_eval  = K_FULL_EVAL

    @property
    def X(self):
        class _ShapeProxy:
            def __init__(self, shape): self.shape = shape
        return _ShapeProxy(self._x_shape)

    def set_mode(self, mode: str):
        assert mode in ("train", "eval")
        self._mode = mode

    def len(self):
        return len(self._offsets)

    def _choose_graph_ids(self, g0: int, g1: int):
        graphs = list(range(g0, g1))
        K = self._K_train if self._mode == "train" else self._K_eval
        if K is not None and len(graphs) > K:
            perm = torch.randperm(len(graphs))[:K]
            graphs = [graphs[j] for j in perm.tolist()]
        return graphs

    def get(self, i: int):
        ci, li = self._offsets[i]
        d = self._cache.get(ci)

        g0 = int(d["seq_ptr"][li].item())
        g1 = int(d["seq_ptr"][li + 1].item())
        graph_ids = self._choose_graph_ids(g0, g1)

        anchors  = d["anchors"]
        edit_pos = int(anchors["edit_pos"][li].item())
        guide_l  = int(anchors["guide_l"][li].item())
        guide_r  = int(anchors["guide_r"][li].item())

        t0, t1 = edit_pos - 5,  edit_pos + 6
        v0, v1 = guide_l  - 1,  guide_r  + 1

        data_list = []
        for g in graph_ids:
            n0 = int(d["graph_ptr"][g].item())
            n1 = int(d["graph_ptr"][g + 1].item())
            e0 = int(d["edge_ptr"][g].item())
            e1 = int(d["edge_ptr"][g + 1].item())

            x   = d["X"][n0:n1].float()
            pos = d["node_pos"][n0:n1].long()
            n   = x.size(0)

            is_edit = (pos == edit_pos).float().unsqueeze(-1)
            is_tgt  = ((pos >= t0) & (pos <= t1)).float().unsqueeze(-1)
            is_var  = ((pos >= v0) & (pos <= v1)).float().unsqueeze(-1)

            idx_norm = (torch.linspace(0., 1., n) if n > 1
                        else torch.zeros(n)).unsqueeze(-1)

            pos_in_tgt = (((pos - t0).float() / max(1, t1 - t0))
                          * is_tgt.squeeze(-1)).unsqueeze(-1)
            pos_in_var = (((pos - v0).float() / max(1, v1 - v0))
                          * is_var.squeeze(-1)).unsqueeze(-1)

            x = torch.cat([x, is_edit, is_tgt, is_var,
                            idx_norm, pos_in_tgt, pos_in_var], dim=1)

            ei = d["edge_index"][:, e0:e1] - n0
            et = d["edge_type"][e0:e1].long()
            data_list.append(Data(x=x, edge_index=ei, edge_type=et))

        out = {"data_list": data_list, "idx": torch.tensor(i, dtype=torch.long)}
        if "y" in d:
            out["y"] = d["y"][li].float()
        return out


class ChunkAwareBatchSampler(Sampler):
    def __init__(self, subset_indices, dataset_offsets, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        chunk_to_indices = defaultdict(list)
        for sub_idx, global_idx in enumerate(subset_indices):
            chunk_id = dataset_offsets[global_idx][0]
            chunk_to_indices[chunk_id].append(sub_idx)
        self._chunk_to_indices = dict(chunk_to_indices)
        self._total = sum(len(v) for v in self._chunk_to_indices.values())

    def __iter__(self):
        chunk_ids = list(self._chunk_to_indices.keys())
        if self.shuffle:
            perm = torch.randperm(len(chunk_ids)).tolist()
            chunk_ids = [chunk_ids[j] for j in perm]

        all_indices = []
        for cid in chunk_ids:
            idxs = self._chunk_to_indices[cid][:]
            if self.shuffle:
                perm = torch.randperm(len(idxs)).tolist()
                idxs = [idxs[j] for j in perm]
            all_indices.extend(idxs)

        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i : i + self.batch_size]

    def __len__(self):
        return (self._total + self.batch_size - 1) // self.batch_size


class ViennaLandscapeDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        d = torch.load(path, map_location="cpu")
        self.X = d["X"]
        self.node_pos = d.get("node_pos", None)
        self.anchors = d.get("anchors", None)
        self.edge_index = d["edge_index"]
        self.edge_type  = d["edge_type"]
        self.graph_ptr = d["graph_ptr"]
        self.edge_ptr  = d["edge_ptr"]
        self.seq_ptr   = d["seq_ptr"]
        self.y = d.get("y", None)
        self._mode = "train"
        self._K_train = K_SUBSAMPLE_TRAIN
        self._K_eval = K_FULL_EVAL

    def set_mode(self, mode: str):
        assert mode in ("train", "eval")
        self._mode = mode

    def len(self):
        return int(self.seq_ptr.numel() - 1)

    def _choose_graph_ids(self, g0: int, g1: int):
        graphs = list(range(g0, g1))
        if self._mode == "train" and self._K_train is not None and len(graphs) > self._K_train:
            perm = torch.randperm(len(graphs))[: self._K_train]
            graphs = [graphs[j] for j in perm.tolist()]
        if self._mode == "eval" and self._K_eval is not None and len(graphs) > self._K_eval:
            perm = torch.randperm(len(graphs))[: self._K_eval]
            graphs = [graphs[j] for j in perm.tolist()]
        return graphs

    def get(self, i: int):
        g0 = int(self.seq_ptr[i].item())
        g1 = int(self.seq_ptr[i + 1].item())
        graph_ids = self._choose_graph_ids(g0, g1)

        if self.node_pos is None or self.anchors is None:
            raise ValueError("This dataset requires node_pos and anchors in the .pt")

        edit_pos = int(self.anchors["edit_pos"][i].item())
        guide_l  = int(self.anchors["guide_l"][i].item())
        guide_r  = int(self.anchors["guide_r"][i].item())

        t0, t1 = edit_pos - 5, edit_pos + 6
        v0, v1 = guide_l - 1, guide_r + 1

        data_list = []
        for g in graph_ids:
            n0 = int(self.graph_ptr[g].item())
            n1 = int(self.graph_ptr[g + 1].item())
            e0 = int(self.edge_ptr[g].item())
            e1 = int(self.edge_ptr[g + 1].item())

            x = self.X[n0:n1].float()
            n = x.size(0)
            pos = self.node_pos[n0:n1].long()

            is_edit = (pos == edit_pos).float().unsqueeze(-1)
            is_tgt  = ((pos >= t0) & (pos <= t1)).float().unsqueeze(-1)
            is_var  = ((pos >= v0) & (pos <= v1)).float().unsqueeze(-1)

            if n > 1:
                idx_norm = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32).unsqueeze(-1)
            else:
                idx_norm = torch.zeros((n, 1), dtype=torch.float32)

            den_t = max(1, (t1 - t0))
            pos_in_tgt = (((pos - t0).float() / float(den_t)) * is_tgt.squeeze(-1)).unsqueeze(-1)
            den_v = max(1, (v1 - v0))
            pos_in_var = (((pos - v0).float() / float(den_v)) * is_var.squeeze(-1)).unsqueeze(-1)

            x = torch.cat([x, is_edit, is_tgt, is_var, idx_norm, pos_in_tgt, pos_in_var], dim=1)
            ei = self.edge_index[:, e0:e1] - n0
            et = self.edge_type[e0:e1].long()
            data_list.append(Data(x=x, edge_index=ei, edge_type=et))

        out = {"data_list": data_list, "idx": torch.tensor(i, dtype=torch.long)}
        if self.y is not None:
            out["y"] = self.y[i].float()
        return out


def collate_sequence_of_graph_sets(batch):
    all_graphs, counts, ys, idxs = [], [], [], []
    for item in batch:
        gs = item["data_list"]
        all_graphs.extend(gs)
        counts.append(len(gs))
        idxs.append(item["idx"])
        if "y" in item:
            ys.append(item["y"])

    pyg_batch = Batch.from_data_list(all_graphs)
    counts_t = torch.tensor(counts, dtype=torch.long)
    seq_ptr = torch.zeros((len(counts) + 1,), dtype=torch.long)
    seq_ptr[1:] = torch.cumsum(counts_t, dim=0)

    out = {"pyg_batch": pyg_batch, "seq_ptr": seq_ptr, "idx": torch.stack(idxs)}
    if ys:
        out["y"] = torch.stack(ys)
    return out


# ── Model (unchanged) ──────────────────────────────────────────────────────

class TypedPairGateMP(MessagePassing):
    def __init__(self, hidden: int):
        super().__init__(aggr="add")
        self.msg_back = nn.Linear(hidden, hidden, bias=False)
        self.msg_pair = nn.Linear(hidden, hidden, bias=False)
        self.upd = nn.Linear(hidden, hidden)
        self.g_src = nn.Linear(hidden, 1, bias=False)
        self.g_dst = nn.Linear(hidden, 1, bias=False)
        self.g_bias = nn.Parameter(torch.zeros(()))

    def forward(self, h, edge_index, edge_type):
        return self.propagate(edge_index, h=h, edge_type=edge_type)

    def message(self, h_i, h_j, edge_type):
        out = torch.empty_like(h_j)
        mask_back = (edge_type == 0)
        if mask_back.any():
            out[mask_back] = self.msg_back(h_j[mask_back])
        mask_pair = (edge_type == 1)
        if mask_pair.any():
            m = self.msg_pair(h_j[mask_pair])
            gate = torch.sigmoid(
                self.g_src(h_j[mask_pair]) + self.g_dst(h_i[mask_pair]) + self.g_bias
            )
            out[mask_pair] = gate * m
        return out

    def update(self, aggr_out, h):
        return torch.relu(h + self.upd(aggr_out))


class PyGLandscapeModel(nn.Module):
    def __init__(self, node_dim: int, hidden: int = 128, layers: int = 3):
        super().__init__()
        self.in_proj = nn.Linear(node_dim, hidden)
        self.mp = nn.ModuleList([TypedPairGateMP(hidden) for _ in range(layers)])
        self.tgt_score = nn.Linear(hidden, 1)
        self.var_score = nn.Linear(hidden, 1)
        self.struct_score = nn.Linear(3 * hidden, 1)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pyg_batch, seq_ptr):
        x0 = pyg_batch.x
        gid = pyg_batch.batch

        h = torch.relu(self.in_proj(x0))
        for layer in self.mp:
            h = layer(h, pyg_batch.edge_index, pyg_batch.edge_type)

        is_edit = x0[:, -EXTRA_NODE_FEATS].unsqueeze(-1)
        is_tgt  = x0[:, -(EXTRA_NODE_FEATS - 1)].unsqueeze(-1)
        is_var  = x0[:, -(EXTRA_NODE_FEATS - 2)].unsqueeze(-1)

        edit_sum = scatter(h * is_edit, gid, dim=0, reduce="sum")
        edit_cnt = scatter(is_edit, gid, dim=0, reduce="sum").clamp_min(1.0)
        edit_g = edit_sum / edit_cnt

        s_t = self.tgt_score(h).squeeze(-1)
        s_t = s_t.masked_fill(is_tgt.squeeze(-1) == 0, -1e9)
        a_t = pyg_softmax(s_t, gid).unsqueeze(-1)
        tgt_g = scatter(h * a_t, gid, dim=0, reduce="sum")

        s_v = self.var_score(h).squeeze(-1)
        s_v = s_v.masked_fill(is_var.squeeze(-1) == 0, -1e9)
        a_v = pyg_softmax(s_v, gid).unsqueeze(-1)
        var_g = scatter(h * a_v, gid, dim=0, reduce="sum")

        g = torch.cat([edit_g, tgt_g, var_g], dim=-1)

        zs = []
        for si in range(seq_ptr.numel() - 1):
            g0 = int(seq_ptr[si].item())
            g1 = int(seq_ptr[si + 1].item())
            gi = g[g0:g1]
            w = torch.softmax(self.struct_score(gi).squeeze(-1), dim=0)
            zs.append((w[:, None] * gi).sum(dim=0))
        Z = torch.stack(zs, dim=0)

        return self.head(Z).squeeze(-1)


# ── Training loops (unchanged) ─────────────────────────────────────────────

def train_one_epoch(model, loader, optim, device, epoch=0):
    model.train()
    mse = nn.MSELoss()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for batch in pbar:
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr = batch["seq_ptr"].to(device)
        y = batch["y"].to(device)
        pred = model(pyg_batch, seq_ptr)
        loss = mse(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * y.numel()
        n += y.numel()
        pbar.set_postfix(loss=total / max(n, 1))
    return total / max(n, 1)


@torch.no_grad()
def eval_one_epoch(model, loader, device, desc="eval", log_every=20):
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for i, batch in enumerate(pbar, 1):
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr = batch["seq_ptr"].to(device)
        y = batch["y"].to(device)
        pred = model(pyg_batch, seq_ptr)
        total += mse(pred, y).detach()
        n += y.numel()
        if (i % log_every) == 0 or i == 1:
            pbar.set_postfix(mse=(total / max(n, 1)).item())
    return (total / max(n, 1)).item()


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    idxs, preds, ys = [], [], []
    for batch in loader:
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr = batch["seq_ptr"].to(device)
        pred = model(pyg_batch, seq_ptr).detach().cpu()
        preds.append(pred)
        idxs.append(batch["idx"].cpu())
        ys.append(batch["y"].cpu())
    return torch.cat(idxs).numpy(), torch.cat(preds).numpy(), torch.cat(ys).numpy()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ds = ChunkedViennaLandscapeDataset("data/training")

    gen = torch.Generator().manual_seed(0)
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    train_sampler = ChunkAwareBatchSampler(
        subset_indices=train_ds.indices,
        dataset_offsets=ds._offsets,
        batch_size=8,
        shuffle=True,
    )
    val_sampler = ChunkAwareBatchSampler(
        subset_indices=val_ds.indices,
        dataset_offsets=ds._offsets,
        batch_size=8,
        shuffle=False,
    )

    train_loader = TorchDataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_sequence_of_graph_sets,
        num_workers=0,
    )
    val_loader = TorchDataLoader(
        val_ds,
        batch_sampler=val_sampler,
        collate_fn=collate_sequence_of_graph_sets,
        num_workers=0,
    )

    node_dim = int(ds.X.shape[1]) + EXTRA_NODE_FEATS
    model = PyGLandscapeModel(node_dim=node_dim, hidden=128, layers=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2, threshold=1e-4, verbose=True
    )

    # ── Resume if a matching checkpoint exists ─────────────────────────────
    train_losses: list[float] = []
    val_losses:   list[float] = []
    start_epoch = 0

    if os.path.isfile(CHECKPOINT_FILE):
        print(f"[checkpoint] Found {CHECKPOINT_FILE}")
        start_epoch, train_losses, val_losses = load_checkpoint(
            CHECKPOINT_FILE, model, optim, scheduler
        )
    else:
        print("[checkpoint] No checkpoint found — starting from scratch.")

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS):
        ds.set_mode("train")
        tr = train_one_epoch(model, train_loader, optim, device, epoch=epoch)

        ds.set_mode("eval")
        va = eval_one_epoch(model, val_loader, device)

        scheduler.step(va)
        train_losses.append(tr)
        val_losses.append(va)
        print(epoch, "train", tr, "val", va)

        # Save after every epoch so a crash loses at most one epoch of work.
        save_checkpoint(epoch, model, optim, scheduler,
                        train_losses, val_losses, CHECKPOINT_FILE)

    # ── Final evaluation & artefacts ──────────────────────────────────────
    idx, pred, y = predict_all(model, val_loader, device)
    order = np.argsort(idx)
    idx, pred, y = idx[order], pred[order], y[order]

    ckpt = {
        "model_state": model.state_dict(),
        "node_dim": node_dim,
        "hidden": 128,
        "layers": 3,
        "extra_node_feats": EXTRA_NODE_FEATS,
    }
    torch.save(ckpt, MODEL_OUTFILE)

    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot
    print("R2:", r2)

    plt.figure()
    plt.plot(idx, y, label="y (true)")
    plt.plot(idx, pred, label="pred")
    plt.xlabel("idx")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.scatter(y, pred, s=12)
    plt.xlabel("y (true)")
    plt.ylabel("pred")
    plt.tight_layout()
    plt.show()

    np.savez(PERFORMANCE_DATA_OUTFILE, idx=idx, pred=pred, y=y)


if __name__ == "__main__":
    main()
