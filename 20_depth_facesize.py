"""
20_depth_facesize.py — STAGE 1 of the depth pipeline: geometric (face-size) distance, scored vs ground truth.

DroneFace full frames (photos_all/) give us real metric ground truth:
    filename  <group>_gp_<H>_eo_<seq>.JPG      group in {ab,cd,ef,gh,ijk}
    height    gp_0=1.5m  gp_3=3.0m  gp_4=4.0m  gp_5=5.0m
    distance  = 17 - seq/2  metres   (seq 00 = 17m ... seq 30 = 2m)

Physics: a face of fixed real width W imaged by a camera of focal length f (px) at distance d
projects to  px = f*W/d   =>   d = k / px   with k = f*W (one constant per camera).
So we FIT k on half the data and TEST on the held-out half.

This script also writes every detected face bbox, which STAGE 3 needs to sample Marigold's depth map
at the same pixels. Run this BEFORE running Marigold.

Outputs:
  reports/depth_facesize.csv   frame, group, height_m, gt_dist_m, x1,y1,x2,y2, px, pred_dist_m, abs_err
  reports/plots/depthfs_*/     predicted-vs-true and error-vs-distance plots
"""
import os
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, csv, datetime
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

FRAMES  = r"C:\dev\train_val\datasets\raw\DnHFaces\open_data_set\photos_all"
OUT_DIR = r"C:\dev\train_val\reports"
HEIGHT_MAP = {'0': 1.5, '3': 3.0, '4': 4.0, '5': 5.0}
DET_SIZE = 1280
SCALES = [1.0, 2.0]        # upscale pass helps the far (small) faces
MIN_DET = 0.30
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
det = app.models['detection']

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

def detect_faces(img):
    """multiscale detect -> list of (x1,y1,x2,y2,score) in ORIGINAL image coords"""
    found = []
    for s in SCALES:
        im = img if s == 1.0 else cv2.resize(img, None, fx=s, fy=s)
        bb, _ = det.detect(im, max_num=0, metric='default')
        if bb is None: continue
        for k in range(len(bb)):
            if bb[k, 4] < MIN_DET: continue
            found.append((*(bb[k, :4] / s), float(bb[k, 4])))
    found.sort(key=lambda f: -f[4])
    kept = []
    for f in found:
        if not any(iou(f[:4], k[:4]) > 0.4 for k in kept): kept.append(f)
    return kept

# ---------- pass 1: detect every face in every frame ----------
files = sorted(f for f in os.listdir(FRAMES) if f.lower().endswith(IMG_EXT))
print(f"[INFO] {len(files)} full frames")
rows, nodet = [], 0
for n, fn in enumerate(files, 1):
    p = fn.split('_')
    if len(p) < 5 or p[1] != 'gp': continue
    group, gp, seq = p[0], p[2], p[4].split('.')[0]
    gt = 17 - int(seq) / 2.0
    img = cv2.imread(os.path.join(FRAMES, fn))
    if img is None: continue
    faces = detect_faces(img)
    if not faces: nodet += 1
    for (x1, y1, x2, y2, sc) in faces:
        px = float(min(x2 - x1, y2 - y1))
        rows.append({'frame': fn, 'group': group, 'height_m': HEIGHT_MAP.get(gp, gp),
                     'gt_dist_m': gt, 'x1': int(x1), 'y1': int(y1), 'x2': int(x2), 'y2': int(y2),
                     'px': round(px, 1), 'det_score': round(sc, 3)})
    if n % 100 == 0: print(f"  ...{n}/{len(files)} frames, {len(rows)} faces")
print(f"[INFO] {len(rows)} faces detected; {nodet} frames with NO detection")

# ---------- detection rate vs distance (detection is the first failure mode) ----------
perframe = defaultdict(lambda: [0, 0])   # gt -> [frames_with_face, frames_total]
seen = set()
for fn in files:
    p = fn.split('_')
    if len(p) < 5 or p[1] != 'gp': continue
    gt = 17 - int(p[4].split('.')[0]) / 2.0
    perframe[gt][1] += 1; seen.add(fn)
hasface = {r['frame'] for r in rows}
for fn in seen:
    p = fn.split('_'); gt = 17 - int(p[4].split('.')[0]) / 2.0
    if fn in hasface: perframe[gt][0] += 1

# ---------- fit d = k/px  (fit on EVEN seq, test on ODD seq -> held out) ----------
def seq_of(fn): return int(fn.split('_')[4].split('.')[0])
fit_rows = [r for r in rows if seq_of(r['frame']) % 2 == 0]
test_rows = [r for r in rows if seq_of(r['frame']) % 2 == 1]
k = float(np.median([r['gt_dist_m'] * r['px'] for r in fit_rows]))     # robust 1-param fit
# 2-param least squares: d = a*(1/px) + b
A = np.array([[1.0 / r['px'], 1.0] for r in fit_rows]); y = np.array([r['gt_dist_m'] for r in fit_rows])
(a2, b2), *_ = np.linalg.lstsq(A, y, rcond=None)
print(f"\n[FIT] 1-param  d = {k:.1f} / px")
print(f"[FIT] 2-param  d = {a2:.1f}/px + {b2:.2f}")

def pred1(px): return k / px
def pred2(px): return a2 / px + b2

for r in rows:
    r['pred_dist_m'] = round(pred1(r['px']), 2)
    r['abs_err'] = round(abs(r['pred_dist_m'] - r['gt_dist_m']), 2)

def report(name, rs, fn_pred):
    if not rs: return
    e = np.array([abs(fn_pred(r['px']) - r['gt_dist_m']) for r in rs])
    rel = np.array([abs(fn_pred(r['px']) - r['gt_dist_m']) / r['gt_dist_m'] for r in rs])
    print(f"  {name:28s} n={len(rs):5d}  MAE={e.mean():5.2f} m  median={np.median(e):5.2f} m  "
          f"MAPE={rel.mean()*100:5.1f}%")

print("\n" + "=" * 70)
print("GEOMETRIC (FACE-SIZE) DEPTH  —  held-out evaluation")
print("=" * 70)
report("1-param, fit set", fit_rows, pred1)
report("1-param, HELD-OUT set", test_rows, pred1)
report("2-param, HELD-OUT set", test_rows, pred2)

print("\n  error by camera height (held-out, 1-param):")
byh = defaultdict(list)
for r in test_rows: byh[r['height_m']].append(abs(pred1(r['px']) - r['gt_dist_m']))
for h in sorted(byh): print(f"    {h:>4}m : MAE {np.mean(byh[h]):5.2f} m   (n={len(byh[h])})")
print("  ^ if MAE grows with height, face-size depth is being confounded by look-down angle")

print("\n  error by true distance (held-out, 1-param):")
byd = defaultdict(list)
for r in test_rows: byd[r['gt_dist_m']].append(abs(pred1(r['px']) - r['gt_dist_m']))
print(f"    {'true(m)':>8} {'MAE(m)':>8} {'n':>5} {'det-rate':>9}")
for d in sorted(byd):
    dr = perframe[d][0] / perframe[d][1] if perframe[d][1] else 0
    print(f"    {d:>8.1f} {np.mean(byd[d]):>8.2f} {len(byd[d]):>5} {dr:>8.0%}")

# ---------- plots ----------
pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("depthfs_%H%M%S"))
os.makedirs(pdir, exist_ok=True)
gt = np.array([r['gt_dist_m'] for r in test_rows]); pr = np.array([pred1(r['px']) for r in test_rows])
fig, ax = plt.subplots(figsize=(7.5, 7))
ax.scatter(gt, pr, s=10, alpha=.3, color='#0C7C88')
lim = [0, 18]; ax.plot(lim, lim, 'k--', lw=1, label='perfect')
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel('true distance (m)'); ax.set_ylabel('predicted (m)')
ax.set_title('Geometric face-size depth vs ground truth (held-out)'); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'pred_vs_true.png'), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ds = sorted(byd)
ax.plot(ds, [np.mean(byd[d]) for d in ds], 'o-', color='#C0392B', lw=2, label='MAE (m)')
ax2 = ax.twinx()
ax2.plot(ds, [perframe[d][0]/perframe[d][1] if perframe[d][1] else 0 for d in ds], 's--',
         color='#7f8c8d', lw=1.5, label='detection rate')
ax2.set_ylabel('frame detection rate'); ax2.set_ylim(0, 1.05)
ax.set_xlabel('true distance (m)'); ax.set_ylabel('MAE (m)')
ax.set_title('Face-size depth error and detection rate vs distance'); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'error_vs_distance.png'), dpi=150); plt.close(fig)

with open(os.path.join(OUT_DIR, "depth_facesize.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\n[SAVED] {os.path.join(OUT_DIR, 'depth_facesize.csv')}  ({len(rows)} faces — STAGE 3 needs this)")
print(f"[SAVED] plots -> {pdir}")
print(f"[FIT CONSTANTS] k={k:.1f}   a={a2:.1f} b={b2:.2f}   (reuse for your DJI after one calibration shot)")
print("[DONE]")
