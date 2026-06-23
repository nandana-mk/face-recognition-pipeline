"""
STEP 1 — FINE-TUNE ArcFace METRIC ADAPTER
==========================================
Strategy: Freeze the pre-trained ArcFace backbone (via DeepFace).
          Train a lightweight linear projection head on top using
          ArcFace/CosFace loss. This adapts the embedding space to
          your specific camera conditions without overfitting.

Architecture:
  ArcFace backbone (frozen) → 512-dim embedding
  → L2-norm
  → Linear(512, EMBED_DIM)   ← trainable adapter
  → L2-norm
  → ArcFace loss head

Output: train_val/models/adapter.pt   (projection weights)
        train_val/models/class_map.json
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

# ── Paths ────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
DATASET_TV  = BASE / "train_val" / "datasets"
MODEL_DIR   = BASE / "train_val" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST    = DATASET_TV / "manifest.json"
ALIGNED_DIR = DATASET_TV / "aligned"

EMBED_DIM   = 256          # projected dimension (lower = less overfit)
EPOCHS      = 40
BATCH_SIZE  = 32
LR          = 3e-4
ARC_S       = 32.0         # ArcFace scale
ARC_M       = 0.50         # ArcFace margin

# ── Check PyTorch ─────────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] PyTorch {torch.__version__} | device: {DEVICE}")
except ImportError:
    print("[ERROR] PyTorch not found. Install with:")
    print("        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

# ── Load DeepFace ArcFace backbone ───────────────────────────────────────────
print("[INFO] Loading ArcFace backbone (DeepFace) …")
os.environ['DEEPFACE_HOME'] = str(BASE)
from deepface import DeepFace

def extract_embedding(img_bgr: np.ndarray) -> np.ndarray:
    """Return 512-dim ArcFace embedding. img_bgr must be 112×112."""
    emb = DeepFace.represent(
        img_path=img_bgr,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False
    )[0]["embedding"]
    v = np.array(emb, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

print("[INFO] Backbone ready\n")

# ── Load manifest + extract embeddings ───────────────────────────────────────
if not MANIFEST.exists():
    print(f"[ERROR] {MANIFEST} not found. Run 00_prepare_data.py first.")
    sys.exit(1)

with open(MANIFEST) as fh:
    manifest = json.load(fh)

train_items = [m for m in manifest if m['split'] == 'train']
identities  = sorted({m['identity'] for m in train_items})
class_map   = {name: idx for idx, name in enumerate(identities)}
NUM_CLASSES = len(identities)

print(f"[INFO] {NUM_CLASSES} identities in training set")
print(f"       {len(train_items)} training images")
print("[INFO] Extracting backbone embeddings (this takes a while) …\n")

EMB_CACHE = MODEL_DIR / "train_embeddings.npz"

if EMB_CACHE.exists():
    print("  [SKIP] Loading cached embeddings …")
    data      = np.load(EMB_CACHE, allow_pickle=True)
    X_train   = data['X']
    y_train   = data['y']
else:
    X_list, y_list = [], []
    skip_count = 0

    for i, item in enumerate(train_items):
        p = Path(item['path'])
        if not p.exists():
            skip_count += 1
            continue

        img = cv2.imread(str(p))
        if img is None:
            skip_count += 1
            continue

        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LANCZOS4)

        emb = extract_embedding(img)
        X_list.append(emb)
        y_list.append(class_map[item['identity']])

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(train_items)} …")

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.int64)
    np.savez(EMB_CACHE, X=X_train, y=y_train)
    if skip_count:
        print(f"  [WARN] Skipped {skip_count} images")

print(f"[INFO] Embeddings ready: {X_train.shape}\n")

# ── ArcFace Loss ──────────────────────────────────────────────────────────────
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=ARC_S, m=ARC_M):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        import math
        self.cos_m  = math.cos(m)
        self.sin_m  = math.sin(m)
        self.th     = math.cos(math.pi - m)
        self.mm     = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine   = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi    = cosine * self.cos_m - sine * self.sin_m
        phi    = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, labels)

# ── Metric Adapter ────────────────────────────────────────────────────────────
class MetricAdapter(nn.Module):
    """Trainable projection head on top of frozen ArcFace backbone."""
    def __init__(self, in_dim=512, out_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)

# ── Dataset ───────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].clone()
        if self.augment:
            # Gaussian noise augmentation in embedding space
            x = x + torch.randn_like(x) * 0.02
            x = F.normalize(x, dim=0)
        return x, self.y[idx]

# ── Training ─────────────────────────────────────────────────────────────────
ds     = EmbeddingDataset(X_train, y_train, augment=True)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

adapter   = MetricAdapter(in_dim=512, out_dim=EMBED_DIM).to(DEVICE)
arc_head  = ArcFaceLoss(in_features=EMBED_DIM, num_classes=NUM_CLASSES).to(DEVICE)

optimizer = torch.optim.AdamW(
    list(adapter.parameters()) + list(arc_head.parameters()),
    lr=LR, weight_decay=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print("=" * 60)
print("TRAINING METRIC ADAPTER")
print(f"  In dim: 512  →  Out dim: {EMBED_DIM}")
print(f"  Classes: {NUM_CLASSES}  |  Epochs: {EPOCHS}")
print(f"  Device: {DEVICE}")
print("=" * 60)

best_loss = float('inf')
for epoch in range(1, EPOCHS + 1):
    adapter.train()
    arc_head.train()

    total_loss = 0.0
    correct    = 0
    total      = 0

    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        proj = adapter(xb)
        loss = arc_head(proj, yb)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * len(yb)
        total      += len(yb)

    scheduler.step()
    avg_loss = total_loss / total

    if epoch % 5 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS} | loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(adapter.state_dict(), MODEL_DIR / "adapter_best.pt")

# Save final + metadata
torch.save(adapter.state_dict(), MODEL_DIR / "adapter_final.pt")

with open(MODEL_DIR / "class_map.json", 'w') as fh:
    json.dump({'class_map': class_map, 'embed_dim': EMBED_DIM,
               'in_dim': 512, 'num_classes': NUM_CLASSES}, fh, indent=2)

print(f"\n[DONE] Best adapter → {MODEL_DIR / 'adapter_best.pt'}")
print(f"       Class map   → {MODEL_DIR / 'class_map.json'}")
print(f"       Best loss   = {best_loss:.4f}")
print("\nRun 02_build_gallery.py next\n")
