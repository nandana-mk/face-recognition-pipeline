"""
07_train_far_adapter.py — train a FAR + DARK robust ADAPTER on insightface embeddings.
Takes video_dataset/<Person>/train crops, makes far (downscale+blur) and dark
(gamma/brightness/noise) augmented copies, embeds them with w600k_r50, and trains a
small adapter (512 -> 256) with ArcFace loss so bright/close and far/dark versions of the
same person land together. The adapter generalizes to NEW people at inference (metric
learning), so it helps Abhinav/Krishna/Tejeswar without them being in training.

Output: train_val/models/far_adapter.pt  +  far_class_map.json
Apply it after get_embedding() at BOTH enrollment and query time.
"""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2, numpy as np, json, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from insightface.app import FaceAnalysis

DS        = r"C:\dev\face-recognition-pipeline\video_dataset"
MODEL_DIR = r"C:\dev\train_val\models"
os.makedirs(MODEL_DIR, exist_ok=True)
CACHE     = os.path.join(MODEL_DIR, "far_train_embeddings.npz")

MAX_PER_PERSON = 300        # cap crops/person before augmentation
AUG_PER        = 4          # far/dark variants per crop (plus the original)
EMBED_DIM      = 256
EPOCHS         = 40
BATCH          = 64
LR             = 3e-4
ARC_S, ARC_M   = 32.0, 0.50
EXT = ('.jpg', '.jpeg', '.png', '.bmp')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] device: {DEVICE}")

# ---------- recognizer (GPU) ----------
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
rec = app.models['recognition']
def emb(img):
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112))
    e = rec.get_feat(img).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

# ---------- far + dark augmentation ----------
def augment_variants(img, n):
    out = [img]
    h, w = img.shape[:2]
    for _ in range(n):
        a = img.copy()
        # FAR: downscale hard then upscale (+ optional blur)
        s = int(np.random.choice([16, 24, 32, 48]))
        a = cv2.resize(a, (s, s), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        if np.random.rand() < 0.5:
            k = int(np.random.choice([3, 5])); a = cv2.GaussianBlur(a, (k, k), 0)
        # DARK: gamma-down + lower brightness/contrast
        g = np.random.uniform(0.4, 1.0)                       # <1 => darker
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)]).astype('uint8')
        a = cv2.LUT(a, lut)
        a = cv2.convertScaleAbs(a, alpha=np.random.uniform(0.6, 1.1), beta=np.random.uniform(-40, 0))
        # sensor noise
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, np.random.uniform(4, 14), a.shape).astype(np.int16)
            a = np.clip(a.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        out.append(a)
    return out

# ---------- build (or load) augmented embeddings ----------
if os.path.exists(CACHE):
    print("[INFO] loading cached embeddings ...")
    d = np.load(CACHE, allow_pickle=True)
    X, y, names = d['X'], d['y'], list(d['names'])
else:
    people = []
    for p in sorted(os.listdir(DS)):
        if p.startswith("_"):
            continue
        if os.path.isdir(os.path.join(DS, p, "train")):
            people.append(p)
    names = people
    name2idx = {n: i for i, n in enumerate(names)}
    X_list, y_list = [], []
    for p in people:
        pdir = os.path.join(DS, p, "train")
        files = [f for f in sorted(os.listdir(pdir)) if f.lower().endswith(EXT)]
        if len(files) > MAX_PER_PERSON:
            step = len(files) / MAX_PER_PERSON
            files = [files[int(i * step)] for i in range(MAX_PER_PERSON)]
        cnt = 0
        for f in files:
            im = cv2.imread(os.path.join(pdir, f))
            if im is None:
                continue
            for v in augment_variants(im, AUG_PER):
                X_list.append(emb(v)); y_list.append(name2idx[p]); cnt += 1
        print(f"  {p:12s}: {len(files)} crops -> {cnt} embeddings (with aug)")
    X = np.array(X_list, dtype=np.float32); y = np.array(y_list, dtype=np.int64)
    np.savez(CACHE, X=X, y=y, names=np.array(names))
    print(f"[INFO] cached -> {CACHE}")

NUM_CLASSES = len(names)
print(f"[INFO] {X.shape[0]} embeddings, {NUM_CLASSES} classes: {', '.join(names)}\n")

# ---------- model ----------
class MetricAdapter(nn.Module):
    def __init__(self, in_dim=512, out_dim=EMBED_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.BatchNorm1d(out_dim))
    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)

class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, num_classes, s=ARC_S, m=ARC_M):
        super().__init__()
        self.s = s; self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        import math
        self.cos_m = math.cos(m); self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m); self.mm = math.sin(math.pi - m) * m
    def forward(self, x, labels):
        cos = F.linear(F.normalize(x), F.normalize(self.weight))
        sin = torch.sqrt((1 - cos.pow(2)).clamp(0, 1))
        phi = cos * self.cos_m - sin * self.sin_m
        phi = torch.where(cos > self.th, phi, cos - self.mm)
        oh = torch.zeros_like(cos); oh.scatter_(1, labels.view(-1, 1), 1)
        out = (oh * phi + (1 - oh) * cos) * self.s
        return F.cross_entropy(out, labels)

class DS_(Dataset):
    def __init__(self, X, y): self.X = torch.from_numpy(X); self.y = torch.from_numpy(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

loader = DataLoader(DS_(X, y), batch_size=BATCH, shuffle=True, drop_last=True)
adapter = MetricAdapter().to(DEVICE)
head = ArcFaceLoss(EMBED_DIM, NUM_CLASSES).to(DEVICE)
opt = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=LR, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

print("=" * 60)
print(f"TRAINING far/dark adapter  512->{EMBED_DIM}  classes={NUM_CLASSES}  device={DEVICE}")
print("=" * 60)
best = 1e9
for ep in range(1, EPOCHS + 1):
    adapter.train(); head.train(); tot = 0.0; nb = 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        loss = head(adapter(xb), yb)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(adapter.parameters(), 5.0); opt.step()
        tot += loss.item(); nb += 1
    sch.step(); avg = tot / max(nb, 1)
    if ep % 5 == 0 or ep == 1:
        print(f"  epoch {ep:3d}/{EPOCHS}  loss={avg:.4f}  lr={sch.get_last_lr()[0]:.2e}")
    if avg < best:
        best = avg; torch.save(adapter.state_dict(), os.path.join(MODEL_DIR, "far_adapter.pt"))

with open(os.path.join(MODEL_DIR, "far_class_map.json"), 'w') as fh:
    json.dump({'names': names, 'embed_dim': EMBED_DIM, 'in_dim': 512}, fh, indent=2)
print(f"\n[DONE] best loss {best:.4f}  ->  {os.path.join(MODEL_DIR, 'far_adapter.pt')}")
print("Next: apply this adapter after get_embedding() at enroll + query, then benchmark.")
