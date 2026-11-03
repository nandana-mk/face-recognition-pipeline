"""
20b_depth_per_identity.py — STAGE 1b: does RECOGNISING the person make DEPTH more accurate?

Stage 1 fit one global constant  d = k/px  (k=243) and hit 0.95 m MAE / 13% relative error.
A ~10% floor that does not shrink with distance looks like per-person bias: real face widths
vary ~+-10%, but we used one k for everybody. Since the pipeline already identifies the person,
we can use THEIR constant instead of the population average.

Protocol (leakage-free):
  identify each detected face (gallery restricted to the subjects the filename says are present)
  FIT   k_global and per-identity k_i on EVEN seq frames
  TEST  both on HELD-OUT ODD seq frames
Reports MAE for global vs per-identity, per-subject constants, and how much error identity explains.

Run AFTER 20_depth_facesize.py.
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
FRAMES = os.path.join(BASE, "photos_all")
PORT   = os.path.join(BASE, "portraits")
TRIO_C = os.path.join(BASE, "trio_cam")
TRIO_G = os.path.join(BASE, "trio_gp")
OUT_DIR = r"C:\dev\train_val\reports"
HEIGHT_MAP = {'0': 1.5, '3': 3.0, '4': 4.0, '5': 5.0}
DET_SIZE, SCALES, MIN_DET = 1280, [1.0, 2.0], 0.30
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
det, rec = app.models['detection'], app.models['recognition']

def unit(img112):
    f = rec.get_feat(img112).flatten().astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-9)

def align_from_photo(path):
    im = cv2.imread(path)
    if im is None: return None
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is not None and len(bb) > 0:
        k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        return face_align.norm_crop(im, kps[k])
    return cv2.resize(im, (112, 112))

def find_frontal(subj):
    for folder, prefix in [(PORT, f"{subj}_"), (TRIO_C, f"{subj}_cam_0_porf"), (TRIO_G, f"{subj}_gp_0_porf")]:
        if not os.path.isdir(folder): continue
        for f in sorted(os.listdir(folder)):
            if f.lower().startswith(prefix) and f.lower().endswith(IMG_EXT):
                return os.path.join(folder, f)
    return None

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

# ---------- gallery ----------
subjects = sorted({f.split('_')[0] for f in os.listdir(os.path.join(BASE, "photos_all_faces"))
                   if f.lower().endswith(IMG_EXT)})
REF = {}
for s in subjects:
    fp = find_frontal(s)
    if fp:
        c = align_from_photo(fp)
        if c is not None: REF[s] = unit(c)
print(f"[INFO] references: {sorted(REF)}")

# ---------- detect + identify every face ----------
files = sorted(f for f in os.listdir(FRAMES) if f.lower().endswith(IMG_EXT))
rows = []
for n, fn in enumerate(files, 1):
    p = fn.split('_')
    if len(p) < 5 or p[1] != 'gp': continue
    group, gp, seq = p[0], p[2], p[4].split('.')[0]
    cands = [c for c in group if c in REF]          # filename says who is in frame
    if not cands: continue
    gt = 17 - int(seq) / 2.0
    img = cv2.imread(os.path.join(FRAMES, fn))
    if img is None: continue
    found = []
    for s in SCALES:
        im = img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s)
        bb, kps = det.detect(im, max_num=0, metric='default')
        if bb is None: continue
        for k in range(len(bb)):
            if bb[k, 4] < MIN_DET: continue
            found.append((bb[k, :4] / s, kps[k] / s, float(bb[k, 4]), s))
    found.sort(key=lambda f: -f[2])
    kept = []
    for f in found:
        if not any(iou(f[0], g[0]) > 0.4 for g in kept): kept.append(f)
    for box, kp, sc, s in kept:
        x1, y1, x2, y2 = box
        px = float(min(x2 - x1, y2 - y1))
        src = img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s)
        e = unit(face_align.norm_crop(src, kp * s))
        sims = {c: float(e @ REF[c]) for c in cands}
        who = max(sims, key=sims.get)
        rows.append({'frame': fn, 'group': group, 'height_m': HEIGHT_MAP.get(gp, gp),
                     'gt_dist_m': gt, 'px': round(px, 1), 'ident': who,
                     'sim': round(sims[who], 3), 'seq': int(seq)})
    if n % 100 == 0: print(f"  ...{n}/{len(files)} frames, {len(rows)} faces")
print(f"[INFO] {len(rows)} faces identified\n")

# ---------- fit on EVEN seq, test on ODD seq ----------
fit = [r for r in rows if r['seq'] % 2 == 0]
test = [r for r in rows if r['seq'] % 2 == 1]
k_global = float(np.median([r['gt_dist_m'] * r['px'] for r in fit]))
k_id = {}
for s in REF:
    v = [r['gt_dist_m'] * r['px'] for r in fit if r['ident'] == s]
    if len(v) >= 5: k_id[s] = float(np.median(v))
print(f"[FIT] global k = {k_global:.1f}")
print(f"[FIT] per-identity k: " + ", ".join(f"{s}={k_id[s]:.0f}" for s in sorted(k_id)))
spread = (max(k_id.values()) - min(k_id.values())) / k_global * 100 if k_id else 0
print(f"[FIT] per-person spread = {spread:.0f}% of k  (this is the bias a global k cannot capture)")

def err(rs, kfn):
    e = np.array([abs(kfn(r) / r['px'] - r['gt_dist_m']) for r in rs])
    rel = np.array([abs(kfn(r) / r['px'] - r['gt_dist_m']) / r['gt_dist_m'] for r in rs])
    return e.mean(), np.median(e), rel.mean() * 100

print("\n" + "=" * 70)
print("GLOBAL k  vs  PER-IDENTITY k   (held-out odd-seq frames)")
print("=" * 70)
for label, kfn in [("global k (Stage 1)", lambda r: k_global),
                   ("per-identity k", lambda r: k_id.get(r['ident'], k_global))]:
    m, md, mape = err(test, kfn)
    print(f"  {label:22s}  MAE={m:5.2f} m   median={md:5.2f} m   rel={mape:5.1f}%")
mg, _, pg = err(test, lambda r: k_global)
mi, _, pi = err(test, lambda r: k_id.get(r['ident'], k_global))
print(f"\n  IMPROVEMENT: MAE {mg:.2f} -> {mi:.2f} m  ({(mg-mi)/mg*100:+.0f}%)   "
      f"relative {pg:.1f}% -> {pi:.1f}%")

# ---------- per-subject breakdown ----------
print("\n  per-identity (held-out):")
print(f"    {'id':>3} {'n':>5} {'k_i':>7} {'MAE global':>11} {'MAE per-id':>11}")
for s in sorted(k_id):
    rs = [r for r in test if r['ident'] == s]
    if not rs: continue
    a, _, _ = err(rs, lambda r: k_global); b, _, _ = err(rs, lambda r: k_id[s])
    print(f"    {s:>3} {len(rs):>5} {k_id[s]:>7.0f} {a:>10.2f}m {b:>10.2f}m")

# ---------- plot ----------
pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("depthid_%H%M%S"))
os.makedirs(pdir, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
for a, kfn, ttl in [(ax[0], lambda r: k_global, f'global k  (MAE {mg:.2f} m)'),
                    (ax[1], lambda r: k_id.get(r['ident'], k_global), f'per-identity k  (MAE {mi:.2f} m)')]:
    gt = [r['gt_dist_m'] for r in test]; pr = [kfn(r) / r['px'] for r in test]
    a.scatter(gt, pr, s=10, alpha=.3, color='#0C7C88')
    a.plot([0, 18], [0, 18], 'k--', lw=1)
    a.set_xlim(0, 18); a.set_ylim(0, 18); a.set_xlabel('true distance (m)'); a.set_title(ttl); a.grid(alpha=.3)
ax[0].set_ylabel('predicted distance (m)')
fig.suptitle('Does knowing WHO it is improve distance? (held-out)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(os.path.join(pdir, 'per_identity_depth.png'), dpi=150)
plt.close(fig)

with open(os.path.join(OUT_DIR, "depth_per_identity.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n[SAVED] {os.path.join(OUT_DIR, 'depth_per_identity.csv')}")
print(f"[SAVED] plot -> {pdir}")
print("[DONE]")
