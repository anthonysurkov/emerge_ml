import torch

CKPT_IN  = "data/models/r270x_z_65536_k64_e100_ckpt.pt"
CKPT_OUT = "data/models/r270x_z_65536_k64_e100_ckpt_infer.pt"

# ── match whatever you used in training ───────────────────────────────────
NODE_DIM        = 11  # <-- fill this in (= ds.X.shape[1] + EXTRA_NODE_FEATS)
HIDDEN          = 128
LAYERS          = 3
EXTRA_NODE_FEATS = 6

# ─────────────────────────────────────────────────────────────────────────
ckpt = torch.load(CKPT_IN, map_location="cpu")

ckpt["node_dim"]        = NODE_DIM
ckpt["hidden"]          = HIDDEN
ckpt["layers"]          = LAYERS
ckpt["extra_node_feats"] = EXTRA_NODE_FEATS

torch.save(ckpt, CKPT_OUT)
print(f"Saved patched ckpt → {CKPT_OUT}")
print(f"  node_dim={NODE_DIM}  hidden={HIDDEN}  layers={LAYERS}  extra_node_feats={EXTRA_NODE_FEATS}")
