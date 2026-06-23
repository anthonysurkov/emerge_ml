"""
eval_landscape.py — standalone evaluation for ViennaLandscape GNN models.

Supports:
  • Binary classification models  (task=binary)   → accuracy, AUC, sigmoid scatter
  • Regression models             (task=regression) → R², pred-vs-true scatter + line plot

Dataset sources:
  • Single .pt file  (--data path/to/file.pt)
  • Chunked directory (--data path/to/chunk_dir  --chunked)

Usage examples
--------------
# Binary / single-file dataset
python eval_landscape.py \
    --model  data/models/r255x_10000_k128_e40_binary_dot20.pt \
    --data   data/training/r255x_subopts_10000_k128_binary_dot20.pt \
    --task   binary \
    --out    evals/r255x_eval.csv

# Regression / chunked dataset
python eval_landscape.py \
    --model   data/models/r270x_z_65536_k64_e100.pt \
    --data    data/training \
    --chunked \
    --task    regression \
    --out     evals/r270x_z_eval.csv

Optional flags
--------------
  --split      val|train|full   which split to evaluate (default: val)
  --seed       int              random seed for train/val split (default: 0)
  --train-frac float            fraction used for training (default: 0.8)
  --batch-size int              loader batch size (default: 8)
  --no-plot                     skip matplotlib display
  --save-plot  path.png         save figure instead of (or as well as) showing it
"""

import argparse
import os
import sys
import gc
import glob
from collections import OrderedDict, defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import random_split, Sampler
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
from torch_geometric.utils import softmax as pyg_softmax

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EXTRA_NODE_FEATS = 6
CHUNK_CACHE_SIZE = 3
K_FULL_EVAL = None          # use all graphs at eval time


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_extra_feats(x, pos, edit_pos, guide_l, guide_r):
    """Append the 6 positional/region extra node features."""
    t0, t1 = edit_pos - 5, edit_pos + 6
    v0, v1 = guide_l - 1,  guide_r + 1

    is_edit = (pos == edit_pos).float().unsqueeze(-1)
    is_tgt  = ((pos >= t0) & (pos <= t1)).float().unsqueeze(-1)
    is_var  = ((pos >= v0) & (pos <= v1)).float().unsqueeze(-1)

    n = x.size(0)
    idx_norm = (torch.linspace(0., 1., n) if n > 1 else torch.zeros(n)).unsqueeze(-1)

    pos_in_tgt = (((pos - t0).float() / max(1, t1 - t0))
                  * is_tgt.squeeze(-1)).unsqueeze(-1)
    pos_in_var = (((pos - v0).float() / max(1, v1 - v0))
                  * is_var.squeeze(-1)).unsqueeze(-1)

    return torch.cat([x, is_edit, is_tgt, is_var, idx_norm, pos_in_tgt, pos_in_var], dim=1)


class ViennaLandscapeDataset(Dataset):
    """Single-file dataset (original format)."""

    def __init__(self, path: str):
        super().__init__()
        d = torch.load(path, map_location="cpu")
        self.X         = d["X"]
        self.node_pos  = d.get("node_pos")
        self.anchors   = d.get("anchors")
        self.edge_index = d["edge_index"]
        self.edge_type  = d["edge_type"]
        self.graph_ptr  = d["graph_ptr"]
        self.edge_ptr   = d["edge_ptr"]
        self.seq_ptr    = d["seq_ptr"]
        # support both "label" (binary) and "y" (regression) keys
        self.y = d.get("y", d.get("label"))
        self._mode   = "eval"

    def set_mode(self, mode: str):
        assert mode in ("train", "eval")
        self._mode = mode

    def len(self):
        return int(self.seq_ptr.numel() - 1)

    def _choose_graph_ids(self, g0, g1):
        graphs = list(range(g0, g1))
        if self._mode == "eval" and K_FULL_EVAL is not None and len(graphs) > K_FULL_EVAL:
            perm = torch.randperm(len(graphs))[:K_FULL_EVAL]
            graphs = [graphs[j] for j in perm.tolist()]
        return graphs

    def get(self, i: int):
        g0 = int(self.seq_ptr[i].item())
        g1 = int(self.seq_ptr[i + 1].item())
        graph_ids = self._choose_graph_ids(g0, g1)

        if self.node_pos is None or self.anchors is None:
            raise ValueError("Dataset .pt must contain node_pos and anchors.")

        edit_pos = int(self.anchors["edit_pos"][i].item())
        guide_l  = int(self.anchors["guide_l"][i].item())
        guide_r  = int(self.anchors["guide_r"][i].item())

        data_list = []
        for g in graph_ids:
            n0 = int(self.graph_ptr[g].item())
            n1 = int(self.graph_ptr[g + 1].item())
            e0 = int(self.edge_ptr[g].item())
            e1 = int(self.edge_ptr[g + 1].item())

            x   = self.X[n0:n1].float()
            pos = self.node_pos[n0:n1].long()
            x   = _build_extra_feats(x, pos, edit_pos, guide_l, guide_r)

            ei = self.edge_index[:, e0:e1] - n0
            et = self.edge_type[e0:e1].long()
            data_list.append(Data(x=x, edge_index=ei, edge_type=et))

        out = {"data_list": data_list, "idx": torch.tensor(i, dtype=torch.long)}
        if self.y is not None:
            out["y"] = self.y[i].float()
        return out


# ── Chunked dataset ────────────────────────────────────────────────────────

class LRUChunkCache:
    def __init__(self, paths, maxsize):
        self._paths  = paths
        self._maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()

    def get(self, chunk_idx):
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

        self._offsets = []
        _x_shape = None
        for ci, p in enumerate(paths):
            tmp = torch.load(p, map_location="cpu")
            n_seqs = int(tmp["seq_ptr"].numel()) - 1
            for li in range(n_seqs):
                self._offsets.append((ci, li))
            if _x_shape is None:
                _x_shape = tmp["X"].shape
            del tmp; gc.collect()

        self._x_shape = _x_shape
        self._cache   = LRUChunkCache(paths, maxsize=CHUNK_CACHE_SIZE)
        self._mode    = "eval"

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

    def _choose_graph_ids(self, g0, g1):
        graphs = list(range(g0, g1))
        if self._mode == "eval" and K_FULL_EVAL is not None and len(graphs) > K_FULL_EVAL:
            perm = torch.randperm(len(graphs))[:K_FULL_EVAL]
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

        data_list = []
        for g in graph_ids:
            n0 = int(d["graph_ptr"][g].item())
            n1 = int(d["graph_ptr"][g + 1].item())
            e0 = int(d["edge_ptr"][g].item())
            e1 = int(d["edge_ptr"][g + 1].item())

            x   = d["X"][n0:n1].float()
            pos = d["node_pos"][n0:n1].long()
            x   = _build_extra_feats(x, pos, edit_pos, guide_l, guide_r)

            ei = d["edge_index"][:, e0:e1] - n0
            et = d["edge_type"][e0:e1].long()
            data_list.append(Data(x=x, edge_index=ei, edge_type=et))

        out = {"data_list": data_list, "idx": torch.tensor(i, dtype=torch.long)}
        y_raw = d.get("y", d.get("label"))
        if y_raw is not None:
            out["y"] = y_raw[li].float()
        return out


# ── Collate ────────────────────────────────────────────────────────────────

def collate_sequence_of_graph_sets(batch):
    all_graphs, counts, ys, idxs = [], [], [], []
    for item in batch:
        all_graphs.extend(item["data_list"])
        counts.append(len(item["data_list"]))
        idxs.append(item["idx"])
        if "y" in item:
            ys.append(item["y"])

    pyg_batch = Batch.from_data_list(all_graphs)
    counts_t  = torch.tensor(counts, dtype=torch.long)
    seq_ptr   = torch.zeros(len(counts) + 1, dtype=torch.long)
    seq_ptr[1:] = torch.cumsum(counts_t, dim=0)

    out = {"pyg_batch": pyg_batch, "seq_ptr": seq_ptr, "idx": torch.stack(idxs)}
    if ys:
        out["y"] = torch.stack(ys)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class TypedPairGateMP(MessagePassing):
    def __init__(self, hidden: int):
        super().__init__(aggr="add")
        self.msg_back = nn.Linear(hidden, hidden, bias=False)
        self.msg_pair = nn.Linear(hidden, hidden, bias=False)
        self.upd      = nn.Linear(hidden, hidden)
        self.g_src    = nn.Linear(hidden, 1, bias=False)
        self.g_dst    = nn.Linear(hidden, 1, bias=False)
        self.g_bias   = nn.Parameter(torch.zeros(()))

    def forward(self, h, edge_index, edge_type):
        return self.propagate(edge_index, h=h, edge_type=edge_type)

    def message(self, h_i, h_j, edge_type):
        out = torch.empty_like(h_j)
        mask_back = edge_type == 0
        if mask_back.any():
            out[mask_back] = self.msg_back(h_j[mask_back])
        mask_pair = edge_type == 1
        if mask_pair.any():
            m    = self.msg_pair(h_j[mask_pair])
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
        self.in_proj      = nn.Linear(node_dim, hidden)
        self.mp           = nn.ModuleList([TypedPairGateMP(hidden) for _ in range(layers)])
        self.tgt_score    = nn.Linear(hidden, 1)
        self.var_score    = nn.Linear(hidden, 1)
        self.struct_score = nn.Linear(3 * hidden, 1)
        self.head         = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, pyg_batch, seq_ptr):
        x0  = pyg_batch.x
        gid = pyg_batch.batch

        h = torch.relu(self.in_proj(x0))
        for layer in self.mp:
            h = layer(h, pyg_batch.edge_index, pyg_batch.edge_type)

        is_edit = x0[:, -EXTRA_NODE_FEATS].unsqueeze(-1)
        is_tgt  = x0[:, -(EXTRA_NODE_FEATS - 1)].unsqueeze(-1)
        is_var  = x0[:, -(EXTRA_NODE_FEATS - 2)].unsqueeze(-1)

        edit_sum = scatter(h * is_edit, gid, dim=0, reduce="sum")
        edit_cnt = scatter(is_edit,     gid, dim=0, reduce="sum").clamp_min(1.0)
        edit_g   = edit_sum / edit_cnt

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
            g0_i = int(seq_ptr[si].item())
            g1_i = int(seq_ptr[si + 1].item())
            gi   = g[g0_i:g1_i]
            w    = torch.softmax(self.struct_score(gi).squeeze(-1), dim=0)
            zs.append((w[:, None] * gi).sum(dim=0))
        return self.head(torch.stack(zs, dim=0)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_all(model, loader, device, task):
    model.eval()
    idxs, preds, ys = [], [], []
    for batch in loader:
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr   = batch["seq_ptr"].to(device)
        raw       = model(pyg_batch, seq_ptr).detach().cpu()
        if task == "binary":
            raw = torch.sigmoid(raw)
        preds.append(raw)
        idxs.append(batch["idx"].cpu())
        if "y" in batch:
            ys.append(batch["y"].cpu())

    idxs  = torch.cat(idxs).numpy()
    preds = torch.cat(preds).numpy()
    ys    = torch.cat(ys).numpy() if ys else None
    order = np.argsort(idxs)
    return idxs[order], preds[order], (ys[order] if ys is not None else None)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def best_f1_threshold(label, pred_p):
    """Return (threshold, f1, precision, recall) that maximises F1."""
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(label, pred_p)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    best_i = f1s[:-1].argmax()
    return thresholds[best_i], f1s[best_i], prec[best_i], rec[best_i]


def plot_binary(idx, pred_p, label, save_path=None, show=True):
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        roc_curve, precision_recall_curve,
    )
    pos_rate = label.mean()
    pred_bin_05 = (pred_p >= 0.5).astype(int)
    acc   = accuracy_score(label, pred_bin_05)
    auc   = roc_auc_score(label, pred_p)
    f1_05 = f1_score(label, pred_bin_05)
    best_t, best_f1, best_prec, best_rec = best_f1_threshold(label, pred_p)
    pred_bin_best = (pred_p >= best_t).astype(int)
    acc_best = accuracy_score(label, pred_bin_best)
    fpr, tpr, _          = roc_curve(label, pred_p)
    prec_c, rec_c, thr_c = precision_recall_curve(label, pred_p)
    f1_curve = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-9)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Binary eval  |  pos rate={pos_rate:.2%}  AUC={auc:.3f}\n"
        f"@ thresh=0.50: acc={acc:.3f}  F1={f1_05:.3f}\n"
        f"@ thresh={best_t:.3f} (best F1): acc={acc_best:.3f}  "
        f"F1={best_f1:.3f}  prec={best_prec:.3f}  rec={best_rec:.3f}",
        fontsize=10,
    )

    # 1. Scatter coloured by true label, both thresholds marked
    ax = axes[0, 0]
    ax.scatter(idx, pred_p, c=label, cmap="bwr", s=10, alpha=0.6)
    ax.axhline(0.5,    linestyle="--", color="gray",      linewidth=1,   label="thresh=0.50")
    ax.axhline(best_t, linestyle="--", color="darkgreen", linewidth=1.2, label=f"thresh={best_t:.3f} (best F1)")
    ax.set_xlabel("Sample index"); ax.set_ylabel("Predicted probability")
    ax.set_title("Predictions (red=positive)"); ax.legend(fontsize=8)

    # 2. ROC curve
    ax = axes[0, 1]
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend()

    # 3. Precision-Recall curve
    ax = axes[1, 0]
    ax.plot(rec_c, prec_c, lw=2, color="steelblue")
    ax.axhline(pos_rate, linestyle=":", color="gray", linewidth=1, label=f"baseline (pos={pos_rate:.2%})")
    ax.scatter([best_rec], [best_prec], color="darkgreen", zorder=5, label=f"best F1={best_f1:.3f} @ t={best_t:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve"); ax.legend(fontsize=8)

    # 4. F1 vs threshold
    ax = axes[1, 1]
    ax.plot(thr_c, f1_curve[:-1], lw=2, color="purple")
    ax.axvline(0.5,    linestyle="--", color="gray",      linewidth=1,   label="thresh=0.50")
    ax.axvline(best_t, linestyle="--", color="darkgreen", linewidth=1.2, label=f"best F1 @ {best_t:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1")
    ax.set_title("F1 vs decision threshold"); ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()




def plot_regression(idx, pred, y, save_path=None, show=True):
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2     = 1.0 - ss_res / ss_tot

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Line: true vs pred over index
    axes[0].plot(idx, y,    label="true", alpha=0.8, lw=1)
    axes[0].plot(idx, pred, label="pred", alpha=0.8, lw=1)
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"True vs Predicted  |  R²={r2:.4f}")
    axes[0].legend()

    # Scatter: true vs pred
    mn, mx = min(y.min(), pred.min()), max(y.max(), pred.max())
    axes[1].scatter(y, pred, s=14, alpha=0.5)
    axes[1].plot([mn, mx], [mn, mx], "k--", lw=1, label="y=x")
    axes[1].set_xlabel("True")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Scatter  |  R²={r2:.4f}")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot] Saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a ViennaLandscape GNN model.")
    p.add_argument("--model",      required=True,  help="Path to model .pt checkpoint.")
    p.add_argument("--data",       required=True,  help="Path to dataset .pt file or chunk directory.")
    p.add_argument("--task",       required=True,  choices=["binary", "regression"],
                   help="binary → BCE + accuracy/AUC; regression → MSE + R².")
    p.add_argument("--out",        default=None,   help="Output CSV path (default: <model_stem>_eval.csv).")
    p.add_argument("--chunked",    action="store_true", help="Load data as chunked directory.")
    p.add_argument("--split",      default="val",  choices=["val", "train", "full"],
                   help="Which split to evaluate (default: val).")
    p.add_argument("--seed",       type=int, default=0,   help="RNG seed for train/val split.")
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--no-plot",    action="store_true")
    p.add_argument("--save-plot",  default=None,   help="Path to save the figure (e.g. out.png).")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # ── Load dataset ────────────────────────────────────────────────────────
    if args.chunked:
        ds = ChunkedViennaLandscapeDataset(args.data)
    else:
        ds = ViennaLandscapeDataset(args.data)

    ds.set_mode("eval")

    n       = len(ds)
    n_train = int(args.train_frac * n)
    n_val   = n - n_train

    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    if args.split == "val":
        eval_ds = val_ds
    elif args.split == "train":
        eval_ds = train_ds
    else:
        eval_ds = ds      # full dataset, no split

    loader = TorchDataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_sequence_of_graph_sets,
        num_workers=0,
    )

    # ── Load model ──────────────────────────────────────────────────────────
    ckpt     = torch.load(args.model, map_location=device)
    node_dim = ckpt["node_dim"]
    hidden   = ckpt.get("hidden", 128)
    layers   = ckpt.get("layers", 3)

    model = PyGLandscapeModel(node_dim=node_dim, hidden=hidden, layers=layers).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[model] Loaded from {args.model}  (node_dim={node_dim}, hidden={hidden}, layers={layers})")

    # ── Predict ─────────────────────────────────────────────────────────────
    idx, pred, y = predict_all(model, loader, device, task=args.task)

    # ── Metrics ─────────────────────────────────────────────────────────────
    if y is not None:
        if args.task == "binary":
            from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
            pred_bin = (pred >= 0.5).astype(int)
            acc  = accuracy_score(y, pred_bin)
            auc  = roc_auc_score(y, pred)
            f1   = f1_score(y, pred_bin)
            best_t, best_f1, best_prec, best_rec = best_f1_threshold(y, pred)
            pred_bin_best = (pred >= best_t).astype(int)
            acc_best = accuracy_score(y, pred_bin_best)
            print(f"  Pos rate : {y.mean():.2%}")
            print(f"  AUC      : {auc:.4f}")
            print(f"  --- threshold = 0.50 ---")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  F1       : {f1:.4f}")
            print(f"  --- threshold = {best_t:.4f} (best F1) ---")
            print(f"  Accuracy : {acc_best:.4f}")
            print(f"  F1       : {best_f1:.4f}")
            print(f"  Precision: {best_prec:.4f}")
            print(f"  Recall   : {best_rec:.4f}")
        else:
            ss_res = ((y - pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2     = 1.0 - ss_res / ss_tot
            rmse   = np.sqrt(((y - pred) ** 2).mean())
            print(f"  R²   : {r2:.4f}")
            print(f"  RMSE : {rmse:.4f}")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    out_csv = args.out
    if out_csv is None:
        stem    = os.path.splitext(os.path.basename(args.model))[0]
        out_csv = f"{stem}_{args.split}_eval.csv"

    os.makedirs(os.path.dirname(out_csv) if os.path.dirname(out_csv) else ".", exist_ok=True)

    df_data = {"idx": idx, "pred": pred}
    if y is not None:
        df_data["true"] = y
        if args.task == "binary":
            best_t, _, _, _ = best_f1_threshold(y, pred)
            df_data["pred_bin_05"]   = (pred >= 0.5).astype(int)
            df_data["pred_bin_best"] = (pred >= best_t).astype(int)
            df_data["best_threshold"] = best_t

    df = pd.DataFrame(df_data)
    df.to_csv(out_csv, index=False)
    print(f"[csv] Saved to {out_csv}  ({len(df)} rows)")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if not args.no_plot or args.save_plot:
        show = not args.no_plot
        if args.task == "binary" and y is not None:
            plot_binary(idx, pred, y, save_path=args.save_plot, show=show)
        elif args.task == "regression" and y is not None:
            plot_regression(idx, pred, y, save_path=args.save_plot, show=show)
        else:
            print("[plot] No labels available — skipping plot.")


if __name__ == "__main__":
    main()
