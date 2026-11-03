"""Audit video_dataset: flag likely-mislabeled and outlier crops, copy them to _review/."""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2, numpy as np, shutil
from insightface.app import FaceAnalysis

DS = r"C:\dev\face-recognition-pipeline\video_dataset"
REVIEW = os.path.join(DS, "_review")
EXT = ('.jpg', '.jpeg', '.png', '.bmp')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
rec = app.models['recognition']

def emb(img):
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112))
    e = rec.get_feat(img).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

people = {}
for person in sorted(os.listdir(DS)):
    if person.startswith("_"):
        continue
    files = []
    for split in ('train', 'test'):
        d = os.path.join(DS, person, split)
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(EXT)]
    if files:
        people[person] = files

print("[INFO] embedding crops ...")
embs = {}
for person, files in people.items():
    arr = []
    for fp in files:
        im = cv2.imread(fp)
        arr.append(None if im is None else emb(im))
    embs[person] = arr
    print(f"  {person:12s}: {len([e for e in arr if e is not None])}/{len(files)}")

cents = {}
for p in embs:
    v = np.mean([e for e in embs[p] if e is not None], axis=0)
    cents[p] = v / (np.linalg.norm(v) + 1e-9)
names = list(cents.keys())
C = np.array([cents[n] for n in names])

os.makedirs(REVIEW, exist_ok=True)
mis = out = 0
print("\n[FLAGS]")
for person, files in people.items():
    for fp, e in zip(files, embs[person]):
        if e is None:
            print("  UNREADABLE", fp); continue
        sims = C @ e
        own = float(cents[person] @ e)
        bi = int(np.argmax(sims)); best, bsim = names[bi], float(sims[bi])
        if best != person and bsim - own > 0.05:
            mis += 1
            print(f"  MISLABEL? {person}/{os.path.basename(fp)} -> looks like {best} ({bsim:.2f} vs own {own:.2f})")
            d = os.path.join(REVIEW, f"{person}__maybe_{best}"); os.makedirs(d, exist_ok=True)
            shutil.copy(fp, os.path.join(d, os.path.basename(fp)))
        elif own < 0.28:
            out += 1
            d = os.path.join(REVIEW, f"{person}__outlier"); os.makedirs(d, exist_ok=True)
            shutil.copy(fp, os.path.join(d, os.path.basename(fp)))

print("\n[TIGHTNESS] (low mean = noisy/mixed folder)")
for p in names:
    es = [e for e in embs[p] if e is not None]
    print(f"  {p:12s} n={len(es):4d}  mean_intra_sim={float(np.mean([cents[p] @ e for e in es])):.3f}")

print(f"\n[SUMMARY] mislabels flagged={mis}  outliers flagged={out}  -> {REVIEW}")
