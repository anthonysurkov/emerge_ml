import torch
import numpy as np
from three_ml import PyGLandscapeModel, ViennaLandscapeDataset, collate_sequence_of_graph_sets
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import os, random

# ── config ────────────────────────────────────────────────────────────────────
MODEL                  = "data/models/r255x_10000_k128_e50_binary.pt"
VAL_DATASET            = "data/r255x_val_subopts.pt"
PERFORMANCE_DATA_OUTFILE = "data/performance/r255x_10000_k128_er0_binary_dot20.npz"

# Set to a float (e.g. 0.05) to binarise labels and pred probabilities.
# Set to None to skip binarisation and work in probability space throughout.
EDIT_CUTOFF = None   # e.g. 0.05

label_map = {
    0: "GGAC", 1: "AC",  2: "ND1",  3: "ND2",  4: "ND3",
    5: "ND4",  6: "ND5", 7: "ND6",  8: "ND7",  9: "ND9",
   10: "ND10",11: "ND11",12: "ND12",13: "ND13",14: "ND14",
}
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(0)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"[info] device = {device}")

ckpt = torch.load(MODEL, map_location="cpu")
print(f"[info] loaded ckpt: {MODEL}")
print(f"[info] node_dim={ckpt['node_dim']} hidden={ckpt['hidden']} "
      f"layers={ckpt['layers']} extra_node_feats={ckpt.get('extra_node_feats', 'NA')}")

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
print(f"[info] n_seqs={len(new_ds)}  base_node_feat_dim={new_ds.X.shape[1]}  "
      f"has_node_pos={new_ds.node_pos is not None}  "
      f"has_anchors={new_ds.anchors is not None}  "
      f"has_y={new_ds.y is not None}")

new_loader = TorchDataLoader(
    new_ds,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_sequence_of_graph_sets,
    num_workers=0,
)


# ── inference ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_with_labels(model, loader, device):
    logits_all, ys, idxs = [], [], []
    pbar = tqdm(loader, desc="predict", leave=True)
    for step, batch in enumerate(pbar, 1):
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr   = batch["seq_ptr"].to(device)

        logit = model(pyg_batch, seq_ptr).detach().cpu()   # raw logits
        prob  = torch.sigmoid(logit)                        # → [0, 1]
        logits_all.append(prob)
        idxs.append(batch["idx"].cpu())
        if "y" in batch:
            ys.append(batch["y"].cpu())

        if step == 1 or step % 10 == 0:
            pbar.set_postfix(
                bs=len(batch["idx"]),
                prob_mean=f"{prob.mean():.4f}",
                prob_min=f"{prob.min():.4f}",
                prob_max=f"{prob.max():.4f}",
            )

    idx  = torch.cat(idxs).numpy()
    pred = torch.cat(logits_all).numpy()          # sigmoid probabilities
    y    = torch.cat(ys).numpy() if ys else None
    return idx, pred, y

idx, pred_prob, y = predict_with_labels(model, new_loader, device)

order = np.argsort(idx)
idx, pred_prob = idx[order], pred_prob[order]
if y is not None:
    y = y[order]

# ── optional binarisation ────────────────────────────────────────────────────
if EDIT_CUTOFF is not None:
    pred_bin = (pred_prob >= EDIT_CUTOFF).astype(float)
    y_bin    = (y         >= EDIT_CUTOFF).astype(float) if y is not None else None
    print(f"\n[info] EDIT_CUTOFF={EDIT_CUTOFF}  "
          f"pred positives={pred_bin.sum():.0f}/{len(pred_bin)}  "
          f"true positives={y_bin.sum():.0f}/{len(y_bin)}")
else:
    pred_bin = None
    y_bin    = None

# ── text summary ─────────────────────────────────────────────────────────────
print(f"\n  n={len(pred_prob)}")
print(f"  pred prob: mean={pred_prob.mean():.4f}  std={pred_prob.std():.4f}  "
      f"min={pred_prob.min():.4f}  max={pred_prob.max():.4f}")

if y is None:
    print("[warn] no labels found in dataset, skipping metrics")
else:
    print(f"  true y:    mean={y.mean():.4f}  std={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}")

    def print_metrics(y_ref, p_ref, label="prob"):
        mse  = float(np.mean((y_ref - p_ref) ** 2))
        mae  = float(np.mean(np.abs(y_ref - p_ref)))
        ss_res = float(np.sum((y_ref - p_ref) ** 2))
        ss_tot = float(np.sum((y_ref - y_ref.mean()) ** 2))
        r2   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        sp   = float(spearmanr(y_ref, p_ref).correlation) if len(y_ref) > 1 else float("nan")
        r    = float(np.corrcoef(y_ref, p_ref)[0, 1])     if len(y_ref) > 1 else float("nan")
        print(f"\n[metrics — {label}]")
        print(f"  MSE={mse:.5f}  RMSE={np.sqrt(mse):.5f}  MAE={mae:.5f}")
        print(f"  R²={r2:.4f}  Pearson r={r:.4f}  Spearman ρ={sp:.4f}")

    print_metrics(y, pred_prob, "probability")
    if pred_bin is not None and y_bin is not None:
        print_metrics(y_bin, pred_bin, f"binary (cutoff={EDIT_CUTOFF})")

    print("\n[head] idx | y | pred_prob | pred_bin | resid")
    for ki in range(min(15, len(y))):
        pb = f"{pred_bin[ki]:.0f}" if pred_bin is not None else "—"
        print(f"  {idx[ki]:4d}  y={y[ki]:.4f}  prob={pred_prob[ki]:.4f}  bin={pb}  "
              f"resid={y[ki]-pred_prob[ki]:+.4f}")

    k = 5
    ranked = list(zip(range(len(y)), y, pred_prob))
    print("\nTop-5 by true y:  ", sorted(ranked, key=lambda t: -t[1])[:k])
    print("Top-5 by pred prob:", sorted(ranked, key=lambda t: -t[2])[:k])


# ── plotting ──────────────────────────────────────────────────────────────────
def _clean_ax(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)

labels = [label_map.get(int(i), str(int(i))) for i in idx]
x = np.arange(len(labels))
w = 0.28

C_TRUE     = "#4DA3D9"   # cyan-blue  — true float
C_PRED     = "#E88FA0"   # salmon     — pred prob
C_TRUE_BIN = "#2176AE"   # darker blue — true binary
C_PRED_BIN = "#C0392B"   # red         — pred binary

plt.rcParams.update({
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
})

n_panels = 2 if (pred_bin is not None and y_bin is not None and y is not None) else 1
fig, axes = plt.subplots(1, n_panels, figsize=(8.0 * n_panels, 2.8), dpi=130)
if n_panels == 1:
    axes = [axes]

# panel 1 — probability view
ax = axes[0]
if y is not None:
    ax.bar(x - w/2, y,         width=w, label="true (rate)",  color=C_TRUE,  edgecolor="none")
ax.bar(x + w/2, pred_prob, width=w, label="pred (prob)",  color=C_PRED,  edgecolor="none")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_ylabel("editing rate / pred probability")
ax.set_title("Probability predictions vs true editing rate", pad=6)
ax.set_ylim(0, max(
    (y.max() if y is not None else 0),
    pred_prob.max()
) * 1.15)
_clean_ax(ax)
ax.legend(frameon=False, ncol=2, loc="upper right")

# panel 2 — binary view (only when EDIT_CUTOFF is set)
if n_panels == 2:
    ax2 = axes[1]
    ax2.bar(x - w/2, y_bin,    width=w, label=f"true ≥{EDIT_CUTOFF}", color=C_TRUE_BIN, edgecolor="none")
    ax2.bar(x + w/2, pred_bin, width=w, label=f"pred ≥{EDIT_CUTOFF}", color=C_PRED_BIN, edgecolor="none")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=35, ha="right")
    ax2.set_ylabel("active (0/1)")
    ax2.set_title(f"Binary predictions (cutoff={EDIT_CUTOFF})", pad=6)
    ax2.set_yticks([0, 1])
    _clean_ax(ax2)
    ax2.legend(frameon=False, ncol=2, loc="upper right")

fig.tight_layout()
plt.show()

# ── save ─────────────────────────────────────────────────────────────────────
save_dict = dict(idx=idx, pred_prob=pred_prob)
if y         is not None: save_dict["y"]        = y
if pred_bin  is not None: save_dict["pred_bin"] = pred_bin
if y_bin     is not None: save_dict["y_bin"]    = y_bin
np.savez(PERFORMANCE_DATA_OUTFILE, **save_dict)
print(f"\n[info] saved → {PERFORMANCE_DATA_OUTFILE}")
