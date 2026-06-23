import torch
import numpy as np
from three_ml import PyGLandscapeModel, ViennaLandscapeDataset, collate_sequence_of_graph_sets
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

import os, random

def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(0)

MODEL = "data/models/r270x_z_65536_k64_e100_ckpt_infer.pt"
VAL_DATASET = "data/proqr_val_subopts.pt"
PERFORMANCE_DATA_OUTFILE = "data/performance/r270x_z_10000_k64_e25_val.pt"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[info] device = {device}")

ckpt = torch.load(MODEL, map_location="cpu")
print(f"[info] loaded ckpt: {MODEL}")
print(f"[info] ckpt node_dim={ckpt['node_dim']} hidden={ckpt['hidden']} layers={ckpt['layers']} extra_node_feats={ckpt.get('extra_node_feats', 'NA')}")

model = PyGLandscapeModel(
    node_dim=ckpt["node_dim"],
    hidden=ckpt["hidden"],
    layers=ckpt["layers"],
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("[info] model loaded + set to eval")

new_ds = ViennaLandscapeDataset(VAL_DATASET)
new_ds.set_mode("eval")
print(f"[info] loaded dataset: {VAL_DATASET}")
print(f"[info] n_seqs={len(new_ds)}  base_node_feat_dim={new_ds.X.shape[1]}  has_node_pos={new_ds.node_pos is not None}  has_anchors={new_ds.anchors is not None}  has_y={new_ds.y is not None}")

new_loader = TorchDataLoader(
    new_ds,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_sequence_of_graph_sets,
    num_workers=0,
)

@torch.no_grad()
def predict_with_labels(model, loader, device):
    preds, ys, idxs = [], [], []
    pbar = tqdm(loader, desc="predict", leave=True)

    for step, batch in enumerate(pbar, 1):
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr = batch["seq_ptr"].to(device)

        pred = model(pyg_batch, seq_ptr).detach().cpu()
        preds.append(pred)

        idxs.append(batch["idx"].cpu())

        if "y" in batch:
            ys.append(batch["y"].cpu())

        # a few live stats
        if step == 1 or step % 10 == 0:
            pbar.set_postfix(
                bs=len(batch["idx"]),
                pred_mean=float(pred.mean()),
                pred_min=float(pred.min()),
                pred_max=float(pred.max()),
            )

    idx = torch.cat(idxs).numpy()
    pred = torch.cat(preds).numpy()

    y = None
    if len(ys) > 0:
        y = torch.cat(ys).numpy()

    return idx, pred, y

idx, pred, y = predict_with_labels(model, new_loader, device)

# order by idx (optional, makes printing nicer)
order = np.argsort(idx)
idx = idx[order]
pred = pred[order]
if y is not None:
    y = y[order]

print(pred)
print(y)

print(f"  n={len(pred)}")
print(f"  pred: mean={pred.mean():.6g} std={pred.std(ddof=0):.6g} min={pred.min():.6g} max={pred.max():.6g}")

if y is None:
    print("[warn] dataset had no labels 'y' in batches, so skipping metrics")
else:
    print("\n[summary] labels")
    print(f"  y: mean={y.mean():.6g} std={y.std(ddof=0):.6g} min={y.min():.6g} max={y.max():.6g}")

    mse = float(np.mean((y - pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y - pred)))

    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    # Pearson r (sometimes nice to print alongside R^2)
    r = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else float("nan")

    print("\n[metrics]")
    print(f"  MSE  = {mse:.6g}")
    print(f"  RMSE = {rmse:.6g}")
    print(f"  MAE  = {mae:.6g}")

    print("\n[head] idx, y, pred, resid")
    for k in range(min(15, len(y))):
        print(f"  {idx[k]:4d}  y={y[k]:.6g}  pred={pred[k]:.6g}  resid={(y[k]-pred[k]):+.6g}")

from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt

# after you have y, pred
sp = spearmanr(y, pred).correlation
print("Spearman:", sp)

# show top-5 by true and by pred
k = 5
print("Top true:", sorted(list(zip(idx, y, pred)), key=lambda t: -t[1])[:k])
print("Top pred:", sorted(list(zip(idx, y, pred)), key=lambda t: -t[2])[:k])

# use raw graph indices directly
idx = idx.astype(int)

# optional sanity check
bad = idx[(idx < 0) | (idx > 95)]
if len(bad) > 0:
    raise ValueError(f"Found indices outside 0..95: {bad}")

labels = [str(i) for i in idx]

x = np.arange(len(idx))
w = 0.38

C_TRUE = "#4DA3D9"
C_PRED = "#E88FA0"

plt.rcParams.update({
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

fig, ax = plt.subplots(figsize=(12.0, 3.2), dpi=130)

ax.bar(x - w/2, y,    width=w, label="true", color=C_TRUE, edgecolor="none")
ax.bar(x + w/2, pred, width=w, label="pred", color=C_PRED, edgecolor="none")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=90, ha="center")

ax.set_xlabel("graph idx")
ax.set_ylabel("editing rate")
ax.set_title("Prediction task: intermolecular reaction context", pad=6)

ax.grid(False)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", length=0)

ax.legend(frameon=False, ncol=2, loc="upper right")

fig.tight_layout()
plt.show()

np.savez(PERFORMANCE_DATA_OUTFILE, idx=idx, pred=pred, y=y)

