"""
STEP 2 — BUILD REFERENCE GALLERY
=================================
Loads the fine-tuned adapter + all TRAIN embeddings.
Computes a per-identity gallery template using:
  - Mean embedding (baseline)
  - Top-K averaging (drop lowest-quality embeddings)

Output: train_val/models/gallery.npz
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path

BASE        = Path(__file__).parent.parent
DATASET_TV  = BASE / "train_val" / "datasets"
MODEL_DIR   = BASE / "train_val" / "models"

MANIFEST    = DATASET_TV / "manifest.json"
EMB_CACHE   = MODEL_DIR / "train_embeddings.npz"
CLASS_MAP_F = MODEL_DIR / "class_map.json"
ADAPTER_PT  = MODEL_DIR / "adapter_best.pt"
GALLERY_OUT = MODEL_DIR / "gallery.npz"

# ── Load adapter ──────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    print("[ERROR] PyTorch required. Run 01_finetune.py first.")
    sys.exit(1)

with open(CLASS_MAP_F) as fh:
    meta = json.load(fh)

class_map  = meta['class_map']
EMBED_DIM  = meta['embed_dim']
IN_DIM     = meta['in_dim']
id2name    = {v: k for k, v in class_map.items()}

class MetricAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )
    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)

adapter = MetricAdapter(IN_DIM, EMBED_DIM).to(DEVICE)
adapter.load_state_dict(torch.load(ADAPTER_PT, map_location=DEVICE))
adapter.eval()
print(f"[INFO] Adapter loaded ({IN_DIM}→{EMBED_DIM})")

def adapt(raw_embs: np.ndarray) -> np.ndarray:
    """Apply adapter to raw backbone embeddings. Input: (N, 512), Output: (N, EMBED_DIM)."""
    with torch.no_grad():
        t = torch.from_numpy(raw_embs).float().to(DEVICE)
        return adapter(t).cpu().numpy()

# ── Load train embeddings ─────────────────────────────────────────────────────
data   = np.load(EMB_CACHE, allow_pickle=True)
X_raw  = data['X']   # (N, 512)
y      = data['y']   # (N,)

with open(MANIFEST) as fh:
    manifest = json.load(fh)
train_items = [m for m in manifest if m['split'] == 'train']

# ── Project embeddings ────────────────────────────────────────────────────────
print("[INFO] Projecting train embeddings through adapter …")
X_proj = adapt(X_raw)   # (N, EMBED_DIM)

# ── Build gallery templates ───────────────────────────────────────────────────
print("[INFO] Building gallery templates …\n")

gallery_embs  = {}    # identity → mean embedding (projected)
gallery_raw   = {}    # identity → mean raw embedding (for raw-space comparison)
gallery_counts= {}

for class_idx in range(meta['num_classes']):
    name  = id2name[class_idx]
    mask  = (y == class_idx)
    if mask.sum() == 0:
        print(f"  [WARN] {name}: no training embeddings")
        continue

    projs = X_proj[mask]    # (K, EMBED_DIM)
    raws  = X_raw[mask]     # (K, 512)

    # Top-K selection: keep embeddings closest to centroid
    centroid = projs.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9

    sims = projs @ centroid                   # cosine sim to centroid
    k    = max(1, int(len(sims) * 0.8))       # keep top 80%
    topk = np.argsort(sims)[-k:]

    mean_proj = projs[topk].mean(axis=0)
    mean_proj /= np.linalg.norm(mean_proj) + 1e-9

    mean_raw  = raws[topk].mean(axis=0)
    mean_raw  /= np.linalg.norm(mean_raw) + 1e-9

    gallery_embs[name]   = mean_proj
    gallery_raw[name]    = mean_raw
    gallery_counts[name] = len(topk)

    print(f"  {name:25s}  {len(topk):4d}/{mask.sum():4d} used  ||emb||={np.linalg.norm(mean_proj):.4f}")

# ── Save gallery ──────────────────────────────────────────────────────────────
names      = list(gallery_embs.keys())
emb_matrix = np.array([gallery_embs[n] for n in names])   # (C, EMBED_DIM)
raw_matrix = np.array([gallery_raw[n]  for n in names])   # (C, 512)

np.savez(GALLERY_OUT,
         names=np.array(names),
         embeddings=emb_matrix,
         raw_embeddings=raw_matrix)

print(f"\n[DONE] Gallery saved → {GALLERY_OUT}")
print(f"       {len(names)} identities | dim={EMBED_DIM}")
print("\nRun 03_validate.py next\n")
