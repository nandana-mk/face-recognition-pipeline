"""
14_droneface_distance.py — recognition-vs-distance benchmark on the DroneFace dataset.

DroneFace naming (this renamed variant):  <id>_gp_<H>_<pose>_<seq>
  id   = subject a..k (11 people)
  gp_H = camera HEIGHT group   -> gp_0=1.5m, gp_3=3m, gp_4=4m, gp_5=5m
  seq  = distance step 00..30  -> distance = 17 - seq/2 metres   (seq00=17m far, seq30=2m near)
        (verified: seq00 mean 30px  ->  seq30 mean 180px)

Protocol (leakage-free, watchlist-style):
  ENROLL each subject from their clean frontal PORTRAIT (the "mugshot").
  TEST  every photos_all_faces crop, grouped by exact distance -> accuracy & mean-sim per metre.
Ground truth is automatic: the crop's leading letter IS the subject.
Output: per-distance table, accuracy/sim-vs-distance plots, CSV.

NOTE: DroneFace has only 11 identities -> use it to MEASURE distance performance, not to TRAIN
(training on 11 IDs overfits, as we saw with the far-adapter).
"""
import os
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, csv, datetime
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.utils import face_align

BASE   = r"C:\dev\train_val\datasets\raw\DnHFaces\open_data_set"
FACES  = os.path.join(BASE, "photos_all_faces")
PORT   = os.path.join(BASE, "portraits")
TRIO_C = os.path.join(BASE, "trio_cam")
TRIO_G = os.path.join(BASE, "trio_gp")
OUT_DIR = r"C:\dev\train_val\reports"
THRESHOLD = 0.40
USE_DENOISE = True                       # match the fr2 deployment pipeline
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
HEIGHT_MAP = {'0': 1.5, '3': 3.0, '4': 4.0, '5': 5.0}

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def embed(img):
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    if USE_DENOISE:
        img = cv2.bilateralFilter(img, 5, 50, 50)
    e = rec.get_feat(img).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

def align_from_photo(path):
    im = cv2.imread(path)
    if im is None: return None
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is not None and len(bb) > 0:
        k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        return face_align.norm_crop(im, kps[k])
    return cv2.resize(im, (112, 112))     # already a tight crop -> use as-is

# ---------- enroll a..k from a clean frontal portrait ----------
def find_frontal(subj):
    for folder, prefix in [(PORT, f"{subj}_"), (TRIO_C, f"{subj}_cam_0_porf"), (TRIO_G, f"{subj}_gp_0_porf")]:
        if not os.path.isdir(folder): continue
        for f in sorted(os.listdir(folder)):
            if f.lower().startswith(prefix) and f.lower().endswith(IMG_EXT):
                return os.path.join(folder, f)
    return None

subjects = sorted({f.split('_')[0] for f in os.listdir(FACES) if f.lower().endswith(IMG_EXT)})
print(f"[INFO] subjects: {subjects}")
names, gal = [], []
for s in subjects:
    fp = find_frontal(s)
    if not fp:
        print(f"  [WARN] no portrait for '{s}' -> not enrolled"); continue
    crop = align_from_photo(fp)
    if crop is None:
        print(f"  [WARN] unreadable portrait for '{s}'"); continue
    names.append(s); gal.append(embed(crop))
    print(f"  [ok] {s} <- {os.path.relpath(fp, BASE)}")
G = np.array(gal, np.float32)
print(f"[INFO] enrolled {len(names)} subjects\n")

# ---------- test every crop, grouped by distance ----------
rows = []
for f in os.listdir(FACES):
    if not f.lower().endswith(IMG_EXT): continue
    p = f.split('_')                       # a_gp_0_ef_00
    if len(p) < 5 or p[1] != 'gp': continue
    sid, gp, seq = p[0], p[2], p[4].split('.')[0]
    if sid not in names: continue
    im = cv2.imread(os.path.join(FACES, f))
    if im is None: continue
    px = min(im.shape[0], im.shape[1])
    e = embed(im)
    sims = G @ e
    i = int(np.argmax(sims)); pred, sim = names[i], float(sims[i])
    dist = 17 - int(seq) / 2.0
    correct = (pred == sid) and (sim >= THRESHOLD)
    rows.append({'subject': sid, 'dist_m': dist, 'height_m': HEIGHT_MAP.get(gp, gp),
                 'px': px, 'pred': pred, 'sim': round(sim, 3),
                 'correct': int(correct), 'accepted': int(sim >= THRESHOLD)})

print(f"[INFO] {len(rows)} test crops evaluated\n")

# ---------- per-distance report ----------
by_d = defaultdict(list)
for r in rows: by_d[r['dist_m']].append(r)
print("=" * 62)
print(f"RECOGNITION vs DISTANCE (DroneFace, enroll=portrait, thr={THRESHOLD})")
print("=" * 62)
print(f"  {'dist(m)':>7} {'n':>4} {'prec':>7} {'recall':>7} {'mean-sim':>9} {'px':>6}")
dists, recs, msims = [], [], []
for d in sorted(by_d):
    b = by_d[d]
    tp = sum(r['correct'] for r in b)          # accepted AND right
    acc_n = sum(r['accepted'] for r in b)      # accepted (sim >= threshold)
    recall = tp / len(b)                       # of all crops, how many correctly IDed
    prec = tp / acc_n if acc_n else 0.0        # of accepted, how many right
    ms = np.mean([r['sim'] for r in b]); mp = np.mean([r['px'] for r in b])
    dists.append(d); recs.append(recall); msims.append(ms)
    print(f"  {d:>7.1f} {len(b):>4} {prec:>7.1%} {recall:>7.1%} {ms:>9.3f} {mp:>6.0f}")

TP = sum(r['correct'] for r in rows); ACC = sum(r['accepted'] for r in rows); N = len(rows)
P = TP / ACC if ACC else 0.0; R = TP / N; F1 = 2 * P * R / (P + R + 1e-9)
good = [d for d, a in zip(dists, recs) if a >= 0.90]
print(f"\n  OVERALL   Precision {P:.3f}   Recall {R:.3f}   F1 {F1:.3f}   (TP={TP} accepted={ACC} n={N})")
print(f"  Reliable range (recall >=90%): to {max(good):.1f} m" if good else "  recall never reaches 90%")

# ---------- per-height ----------
by_h = defaultdict(list)
for r in rows: by_h[r['height_m']].append(r)
print("\n  by camera height:")
for h in sorted(by_h):
    b = by_h[h]; print(f"   {h}m : acc {np.mean([r['correct'] for r in b]):.1%}  (n={len(b)})")

# ---------- plots ----------
pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("droneface_%H%M%S"))
os.makedirs(pdir, exist_ok=True)
plt.figure(figsize=(9, 5))
plt.plot(dists, recs, 'o-', color='#0C7C88', lw=2, label='recall')
plt.plot(dists, msims, 's--', color='#B67611', lw=1.5, label='mean similarity')
plt.axhline(THRESHOLD, color='gray', ls=':', alpha=0.6, label=f'threshold {THRESHOLD}')
plt.xlabel('distance (metres)'); plt.ylabel('accuracy / similarity'); plt.ylim(0, 1.02)
plt.title('DroneFace: recognition vs distance (portrait-enrolled)')
plt.grid(alpha=0.3); plt.legend()   # x-axis: 2 m (near) left -> 17 m (far) right
plt.tight_layout(); plt.savefig(os.path.join(pdir, 'accuracy_vs_distance.png'), dpi=150); plt.close()

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "droneface_distance.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n[SAVED] plot -> {pdir}\\accuracy_vs_distance.png")
print(f"[SAVED] per-crop CSV -> {os.path.join(OUT_DIR, 'droneface_distance.csv')}")
print("[DONE]")
