import torch
import torch.nn as nn
from torch.utils.data import random_split
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as pyg_softmax
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

NUM_SEQS = 10000
NUM_EPOCHS = 40 # R255X seems to plateau around ~40 epochs
NUM_SUBOPTS = 128

PT_DATA_INFILE = f"data/training/r255x_subopts_{NUM_SEQS}_k{NUM_SUBOPTS}_binary_dot20.pt"
MODEL_OUTFILE = f"data/models/r255x_{NUM_SEQS}_k{NUM_SUBOPTS}_e{NUM_EPOCHS}_binary_dot20.pt"
PERFORMANCE_DATA_OUTFILE = f"data/performance/r255x_{NUM_SEQS}_k{NUM_SUBOPTS}_e{NUM_EPOCHS}_binary_dot20.npz"
# (num seqs)_(num subopts)_(num epochs).pt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# extra node features:
# is_edit, is_tgt, is_var, idx_norm, pos_in_tgt, pos_in_var  => 6
EXTRA_NODE_FEATS = 6

# K handling
K_SUBSAMPLE_TRAIN = 64    # <= 256; used only during training (via ds.set_mode)
K_FULL_EVAL = None        # None => use all available per sequence

def bce_loss(logit, label):
    return F.binary_cross_entropy_with_logits(logit, label)

class ViennaLandscapeDataset(Dataset):
    def __init__(self, path: str):
        super().__init__()
        d = torch.load(path, map_location="cpu")
        self.X = d["X"]
        self.node_pos = d.get("node_pos", None)   # NEW
        self.anchors = d.get("anchors", None)     # OPTIONAL NEW (dict: edit_pos, guide_l, guide_r)

        self.edge_index = d["edge_index"]
        self.edge_type  = d["edge_type"]
        self.graph_ptr = d["graph_ptr"]
        self.edge_ptr  = d["edge_ptr"]
        self.seq_ptr   = d["seq_ptr"]
        self.label = d["label"]

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

        # define windows in COMPACT coords
        t0, t1 = edit_pos - 5, edit_pos + 6      # around edit
        v0, v1 = guide_l - 1, guide_r + 1        # guide/N10 window

        data_list = []
        for g in graph_ids:
            n0 = int(self.graph_ptr[g].item())
            n1 = int(self.graph_ptr[g + 1].item())
            e0 = int(self.edge_ptr[g].item())
            e1 = int(self.edge_ptr[g + 1].item())

            x = self.X[n0:n1].float()
            n = x.size(0)

            pos = self.node_pos[n0:n1].long()    # compact coords for these nodes

            # ---- masks from anchors (absolute compact coords)
            is_edit = (pos == edit_pos).float().unsqueeze(-1)
            is_tgt  = ((pos >= t0) & (pos <= t1)).float().unsqueeze(-1)
            is_var  = ((pos >= v0) & (pos <= v1)).float().unsqueeze(-1)

            # ---- position features
            if n > 1:
                idx_norm = torch.linspace(0.0, 1.0, steps=n, dtype=torch.float32).unsqueeze(-1)
            else:
                idx_norm = torch.zeros((n, 1), dtype=torch.float32)

            # 0..1 inside each region, 0 outside (vectorized)
            den_t = max(1, (t1 - t0))
            pos_in_tgt = (((pos - t0).float() / float(den_t)) * is_tgt.squeeze(-1)).unsqueeze(-1)

            den_v = max(1, (v1 - v0))
            pos_in_var = (((pos - v0).float() / float(den_v)) * is_var.squeeze(-1)).unsqueeze(-1)

            x = torch.cat([x, is_edit, is_tgt, is_var, idx_norm, pos_in_tgt, pos_in_var], dim=1)

            ei = self.edge_index[:, e0:e1] - n0
            et = self.edge_type[e0:e1].long()

            data_list.append(Data(x=x, edge_index=ei, edge_type=et))

        out = {"data_list": data_list, "idx": torch.tensor(i, dtype=torch.long)}
        if self.label is not None:
            out["label"] = self.label[i].float()
        return out

def collate_sequence_of_graph_sets(batch):
    all_graphs, counts, labels, idxs = [], [], [], []
    for item in batch:
        gs = item["data_list"]
        all_graphs.extend(gs)
        counts.append(len(gs))
        idxs.append(item["idx"])
        if "label" in item:
            labels.append(item["label"])

    pyg_batch = Batch.from_data_list(all_graphs)

    counts_t = torch.tensor(counts, dtype=torch.long)
    seq_ptr = torch.zeros((len(counts) + 1,), dtype=torch.long)
    seq_ptr[1:] = torch.cumsum(counts_t, dim=0)

    out = {"pyg_batch": pyg_batch, "seq_ptr": seq_ptr, "idx": torch.stack(idxs)}
    if labels:
        out["label"] = torch.stack(labels)
    return out


class TypedPairGateMP(MessagePassing):
    """
    Typed message passing with *sigmoid gating on base-pair edges only*.

    Backbone edges: m = W_back h_j
    Pair edges:     m = sigma(score(h_i, h_j, type=pair)) * W_pair h_j

    No softmax competition. Gate is per-edge (0..1).
    """
    def __init__(self, hidden: int):
        super().__init__(aggr="add")
        self.msg_back = nn.Linear(hidden, hidden, bias=False)
        self.msg_pair = nn.Linear(hidden, hidden, bias=False)
        self.upd = nn.Linear(hidden, hidden)

        # gate network (pair edges only)
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
            gate = torch.sigmoid(self.g_src(h_j[mask_pair]) + self.g_dst(h_i[mask_pair]) + self.g_bias)  # (E_pair,1)
            out[mask_pair] = gate * m

        return out

    def update(self, aggr_out, h):
        return torch.relu(h + self.upd(aggr_out))


class PyGLandscapeModel(nn.Module):
    """
    Per-structure:
      - GNN over all nodes (with pair-edge gates)
      - pool 3 region summaries: edit node, target-side (19..30) attention, var-side (61..73) attention
    Per-sequence:
      - attention over K structure embeddings
      - MLP -> scalar
    """
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

        # node embeddings
        h = torch.relu(self.in_proj(x0))
        for layer in self.mp:
            h = layer(h, pyg_batch.edge_index, pyg_batch.edge_type)

        # appended order: is_edit, is_tgt, is_var, idx_norm, pos_in_tgt, pos_in_var
        is_edit = x0[:, -EXTRA_NODE_FEATS].unsqueeze(-1)
        is_tgt  = x0[:, -(EXTRA_NODE_FEATS - 1)].unsqueeze(-1)
        is_var  = x0[:, -(EXTRA_NODE_FEATS - 2)].unsqueeze(-1)

        # edit pooling
        edit_sum = scatter(h * is_edit, gid, dim=0, reduce="sum")
        edit_cnt = scatter(is_edit, gid, dim=0, reduce="sum").clamp_min(1.0)
        edit_g = edit_sum / edit_cnt

        #tgt-region node attention (19..30)
        s_t = self.tgt_score(h).squeeze(-1)
        s_t = s_t.masked_fill(is_tgt.squeeze(-1) == 0, -1e9)
        a_t = pyg_softmax(s_t, gid).unsqueeze(-1)          # (N,1)
        tgt_g = scatter(h * a_t, gid, dim=0, reduce="sum") # (num_graphs, hidden)

        # var-region node attention (61..73)
        s_v = self.var_score(h).squeeze(-1)
        s_v = s_v.masked_fill(is_var.squeeze(-1) == 0, -1e9)
        a_v = pyg_softmax(s_v, gid).unsqueeze(-1)
        var_g = scatter(h * a_v, gid, dim=0, reduce="sum")

        g = torch.cat([edit_g, tgt_g, var_g], dim=-1)  # (num_graphs, 3*hidden)

        # attention pool graphs -> per-sequence embedding
        zs = []
        for si in range(seq_ptr.numel() - 1):
            g0 = int(seq_ptr[si].item())
            g1 = int(seq_ptr[si + 1].item())
            gi = g[g0:g1]
            w = torch.softmax(self.struct_score(gi).squeeze(-1), dim=0)
            zs.append((w[:, None] * gi).sum(dim=0))
        Z = torch.stack(zs, dim=0)

        return self.head(Z).squeeze(-1)


def train_one_epoch(model, loader, optim, device, epoch=0):
    model.train()
    mse = nn.MSELoss()
    total = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
    for batch in pbar:
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr = batch["seq_ptr"].to(device)

        label = batch["label"].to(device)
        pred  = model(pyg_batch, seq_ptr)
        loss  = bce_loss(pred, label)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item() * label.numel()
        n += label.numel()
        pbar.set_postfix(loss=total / max(n, 1))
    return total / max(n, 1)

@torch.no_grad()
def eval_one_epoch(model, loader, device, desc="eval", log_every=20):
    model.eval()
    mse = nn.MSELoss(reduction="sum")  # sum so we can accumulate exactly
    total = 0.0
    n = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for i, batch in enumerate(pbar, 1):
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr = batch["seq_ptr"].to(device)
        label = batch["label"].to(device)

        pred  = model(pyg_batch, seq_ptr)
        # accumulate without syncing each time
        total += bce_loss(pred, label).detach()   # tensor on device
        n += label.numel()

        if (i % log_every) == 0 or i == 1:
            pbar.set_postfix(mse=(total / max(n, 1)).item())  # sync occasionally

    return (total / max(n, 1)).item()

@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    idxs, preds, labels = [], [], []
    for batch in loader:
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr = batch["seq_ptr"].to(device)
        pred = model(pyg_batch, seq_ptr).detach().cpu()
        idxs.append(batch["idx"].cpu())
        preds.append(torch.sigmoid(pred).cpu())
        labels.append(batch["label"].cpu())

    return torch.cat(idxs).numpy(), torch.cat(preds).numpy(), torch.cat(labels).numpy()


def main():
    ds = ViennaLandscapeDataset(PT_DATA_INFILE)

    gen = torch.Generator().manual_seed(0)
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    train_loader = TorchDataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_sequence_of_graph_sets,
        num_workers=2,
        persistent_workers=True
    )
    val_loader = TorchDataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_sequence_of_graph_sets,
        num_workers=2,
        persistent_workers=True
    )

    node_dim = int(ds.X.shape[1]) + EXTRA_NODE_FEATS
    model = PyGLandscapeModel(node_dim=node_dim, hidden=128, layers=3).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=0.5, patience=2, threshold=1e-4, verbose=True
    )

    for epoch in range(NUM_EPOCHS):
        ds.set_mode("train")
        tr = train_one_epoch(model, train_loader, optim, device, epoch=epoch)
        ds.set_mode("eval")
        va = eval_one_epoch(model, val_loader, device)
        scheduler.step(va)
        print(epoch, "train", tr, "val", va)

    idx, pred_p, label = predict_all(model, val_loader, device)
    order = np.argsort(idx)
    idx, pred_p, label = idx[order], pred_p[order], label[order]
    
    ckpt = {
        "model_state": model.state_dict(),
        "node_dim": node_dim,
        "hidden": 128,
        "layers": 3,
        "extra_node_feats": EXTRA_NODE_FEATS,
    }
    torch.save(ckpt, MODEL_OUTFILE)

    pred_bin = (pred_p >= 0.5).astype(int)
    print("Accuracy:", accuracy_score(label, pred_bin))
    print("AUC:     ", roc_auc_score(label, pred_p))
    
    plt.figure()
    plt.scatter(range(len(label)), pred_p, c=label, cmap="bwr", s=12)
    plt.axhline(0.5, linestyle="--", color="gray")
    plt.xlabel("idx"); plt.ylabel("pred p"); plt.tight_layout(); plt.show()

    np.savez(PERFORMANCE_DATA_OUTFILE, idx=idx, pred=pred_p, label=label)

if __name__ == "__main__":
    main()
