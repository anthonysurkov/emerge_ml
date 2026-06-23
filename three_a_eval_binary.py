import os, random
import numpy as np
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from three_ml_binary import (
    PyGLandscapeModel,
    ViennaLandscapeDataset,
    collate_sequence_of_graph_sets,
)

def seed_everything(seed=0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(0)

MODEL = "data/models/r255x_10000_k128_e40_binary_dot20.pt"
DATASET = "data/training/r255x_subopts_10000_k128_binary_dot20.pt"
OUTFILE = "data/performance/r255x_10000_k128_e40_binary_dot20_loaded_eval.npz"

THRESHOLD = 0.5
BATCH_SIZE = 8

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")
print(f"[info] device = {device}")

ckpt = torch.load(MODEL, map_location="cpu")
print(f"[info] loaded ckpt: {MODEL}")
print(
    f"[info] ckpt node_dim={ckpt['node_dim']} "
    f"hidden={ckpt['hidden']} "
    f"layers={ckpt['layers']} "
    f"extra_node_feats={ckpt.get('extra_node_feats', 'NA')}"
)

model = PyGLandscapeModel(
    node_dim=ckpt["node_dim"],
    hidden=ckpt["hidden"],
    layers=ckpt["layers"],
).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print("[info] model loaded + set to eval")

# ---- recreate exact split
ds = ViennaLandscapeDataset(DATASET)
print(f"[info] loaded dataset: {DATASET}")
print(f"[info] total sequences = {len(ds)}")

gen = torch.Generator().manual_seed(0)
n_total = len(ds)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=gen)

# IMPORTANT: underlying dataset is shared by both Subsets
ds.set_mode("eval")

val_loader = TorchDataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_sequence_of_graph_sets,
    num_workers=0,
)

for batch in val_loader:
    print(batch.keys())
    break

import torch
d = torch.load("data/training/r255x_subopts_10000_k128_binary_dot20.pt", map_location="cpu")
print(d.keys())
        
print(f"[info] train size = {len(train_ds)}")
print(f"[info] val size   = {len(val_ds)}")

@torch.no_grad()
def predict_binary(model, loader, device):
    idxs, logits, probs, labels = [], [], [], []

    pbar = tqdm(loader, desc="predict", leave=True)
    for step, batch in enumerate(pbar, 1):
        pyg_batch = batch["pyg_batch"].to(device)
        seq_ptr = batch["seq_ptr"].to(device)

        pred_logit = model(pyg_batch, seq_ptr).detach().cpu()
        pred_prob = torch.sigmoid(pred_logit)

        logits.append(pred_logit)
        probs.append(pred_prob)
        idxs.append(batch["idx"].cpu())
        labels.append(batch["label"].cpu())

        if step == 1 or step % 10 == 0:
            pbar.set_postfix(
                bs=len(batch["idx"]),
                logit_mean=float(pred_logit.mean()),
                p_mean=float(pred_prob.mean()),
                p_min=float(pred_prob.min()),
                p_max=float(pred_prob.max()),
            )

    idx = torch.cat(idxs).numpy()
    pred_logit = torch.cat(logits).numpy()
    pred_p = torch.cat(probs).numpy()
    label = torch.cat(labels).numpy()

    return idx, pred_logit, pred_p, label

idx, pred_logit, pred_p, label = predict_binary(model, val_loader, device)

# sort by original dataset idx
order = np.argsort(idx)
idx = idx[order]
pred_logit = pred_logit[order]
pred_p = pred_p[order]
label = label[order]

pred_bin = (pred_p >= THRESHOLD).astype(int)
label_int = label.astype(int)

acc = accuracy_score(label_int, pred_bin)
auc = roc_auc_score(label_int, pred_p)
prec = precision_score(label_int, pred_bin, zero_division=0)
rec = recall_score(label_int, pred_bin, zero_division=0)
f1 = f1_score(label_int, pred_bin, zero_division=0)

tn, fp, fn, tp = confusion_matrix(label_int, pred_bin, labels=[0, 1]).ravel()

specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")

print("\n[metrics]")
print(f"  threshold    = {THRESHOLD:.3f}")
print(f"  n            = {len(label_int)}")
print(f"  positives    = {label_int.sum()}")
print(f"  negatives    = {(label_int == 0).sum()}")

print(f"\n  Accuracy     = {acc:.6f}")
print(f"  AUC          = {auc:.6f}")
print(f"  Precision    = {prec:.6f}")
print(f"  Recall       = {rec:.6f}")
print(f"  F1           = {f1:.6f}")
print(f"  Specificity  = {specificity:.6f}")
print(f"  NPV          = {npv:.6f}")
print(f"  FPR          = {fpr:.6f}")
print(f"  FNR          = {fnr:.6f}")

print("\n[confusion matrix]")
print(f"  TP = {tp}")
print(f"  FP = {fp}")
print(f"  TN = {tn}")
print(f"  FN = {fn}")

print("\n[head] idx, label, pred_p, pred_logit, pred_bin")
for j in range(min(20, len(label_int))):
    print(
        f"  {idx[j]:5d}  "
        f"y={label_int[j]}  "
        f"p={pred_p[j]:.6g}  "
        f"logit={pred_logit[j]:+.6g}  "
        f"yhat={pred_bin[j]}"
    )

# inspect mistakes
fp_rows = [(idx[i], label_int[i], pred_p[i], pred_logit[i]) for i in range(len(idx)) if label_int[i] == 0 and pred_bin[i] == 1]
fn_rows = [(idx[i], label_int[i], pred_p[i], pred_logit[i]) for i in range(len(idx)) if label_int[i] == 1 and pred_bin[i] == 0]

print("\nTop false positives:")
for row in sorted(fp_rows, key=lambda t: -t[2])[:10]:
    print(f"  idx={row[0]} y={row[1]} p={row[2]:.6g} logit={row[3]:+.6g}")

print("\nTop false negatives:")
for row in sorted(fn_rows, key=lambda t: t[2])[:10]:
    print(f"  idx={row[0]} y={row[1]} p={row[2]:.6g} logit={row[3]:+.6g}")

# optional threshold sweep
ths = np.linspace(0.0, 1.0, 201)
f1s, precisions, recalls = [], [], []
for t in ths:
    yhat = (pred_p >= t).astype(int)
    f1s.append(f1_score(label_int, yhat, zero_division=0))
    precisions.append(precision_score(label_int, yhat, zero_division=0))
    recalls.append(recall_score(label_int, yhat, zero_division=0))

best_i = int(np.argmax(f1s))
best_t = float(ths[best_i])
best_f1 = float(f1s[best_i])

print(f"\n[threshold sweep]")
print(f"  best threshold by F1 = {best_t:.3f}")
print(f"  best F1             = {best_f1:.6f}")

# plots
plt.figure(figsize=(5.0, 3.2), dpi=130)
plt.scatter(range(len(label_int)), pred_p, c=label_int, cmap="bwr", s=12)
plt.axhline(THRESHOLD, linestyle="--", color="gray")
plt.xlabel("sorted val-set sample")
plt.ylabel("predicted probability")
plt.tight_layout()
plt.show()

plt.figure(figsize=(4.2, 3.2), dpi=130)
plt.plot(ths, f1s, label="F1")
plt.plot(ths, precisions, label="Precision")
plt.plot(ths, recalls, label="Recall")
plt.axvline(best_t, linestyle="--", color="gray")
plt.xlabel("threshold")
plt.ylabel("metric")
plt.legend(frameon=False)
plt.tight_layout()
plt.show()

np.savez(
    OUTFILE,
    idx=idx,
    pred_logit=pred_logit,
    pred_p=pred_p,
    label=label_int,
    pred_bin=pred_bin,
    threshold=np.array([THRESHOLD]),
    accuracy=np.array([acc]),
    auc=np.array([auc]),
    precision=np.array([prec]),
    recall=np.array([rec]),
    f1=np.array([f1]),
    tp=np.array([tp]),
    fp=np.array([fp]),
    tn=np.array([tn]),
    fn=np.array([fn]),
    threshold_grid=ths,
    f1_grid=np.array(f1s),
    precision_grid=np.array(precisions),
    recall_grid=np.array(recalls),
)
print(f"\n[info] saved performance data to: {OUTFILE}")
