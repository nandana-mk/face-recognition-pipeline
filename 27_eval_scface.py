"""
27_eval_scface.py — head-to-head validation on SCface: OLD (raw cosine) vs NEW (AS-Norm).

    python 27_eval_scface.py "C:\\path\\to\\SCface"

WHY SCFACE
  High-resolution frontal MUGSHOT + real surveillance probes at 1.0 / 2.6 / 4.2 m from 5 cameras.
  That is exactly our deployment shape: ONE clean reference photo per person, degraded probes at
  known distance - but with REAL sensor degradation instead of our synthetic kind.

PROTOCOL (open-set, the honest one)
  ENROLL    N subjects, ONE mugshot each              -> the watchlist
  PROBE     all surveillance images of those subjects -> should be recognised   (recall)
  STRANGER  all surveillance images of the HELD-OUT subjects -> must ALL be rejected  (precision)
  Strangers come from the SAME cameras and distances as the probes, so the false-accept rate is
  measured against realistic look-alikes rather than LFW celebrities.

THREE CONFIGURATIONS COMPARED
  A  OLD : raw cosine >= THRESHOLD                    (what you run today)
  B  NEW : AS-Norm z >= z_thresh from cohort.npz      (does the DroneFace calibration TRANSFER?)
  C  NEW : AS-Norm z re-tuned on SCface at A's FAR    (equal-precision ceiling - the fair comparison)

Comparing A and B at their own thresholds is NOT a fair test (different scales). C is the one that
answers "is AS-Norm actually better", because it holds stranger false-accepts equal.

Outputs: console tables (overall, by distance, by camera, by face px) + reports/scface_eval.csv
"""
import os, sys, re, glob, csv
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, datetime
from collections import defaultdict
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.utils import face_align

COHORT_NPZ = r"C:\dev\train_val\models\cohort.npz"
OUT_DIR    = r"C:\dev\train_val\reports"
THRESHOLD  = 0.40          # the OLD model's operating point
N_ENROLL   = 100           # enrolled subjects; the rest become strangers
USE_IR     = False         # cam6-8 are infrared; off by default (different spectrum)
IMG_EXT    = ('.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG')
SEED       = 7
# SCface distance codes -> metres. VERIFIED EMPIRICALLY BELOW via mean face size.
DIST_MAP   = {'1': 4.20, '2': 2.60, '3': 1.00}

rng = np.random.default_rng(SEED)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def embed_batch(c):
    f = rec.get_feat(c); f = np.asarray(f, np.float32).reshape(len(c), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
def detect_align(im):
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0: return None, 0.0
    j = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
    px = float(min(bb[j, 2]-bb[j, 0], bb[j, 3]-bb[j, 1]))
    return face_align.norm_crop(im, kps[j]), px

root = sys.argv[1] if len(sys.argv) > 1 else None
if not root or not os.path.isdir(root):
    print("usage: python 27_eval_scface.py <SCface root folder>"); raise SystemExit

# ---------------- discover ----------------
all_files = [p for p in glob.glob(os.path.join(root, '**', '*'), recursive=True)
             if p.lower().endswith(tuple(e.lower() for e in IMG_EXT))]
mug, surv = {}, []
re_surv = re.compile(r'(\d{3})_cam(\d+)_(\d)', re.I)
re_mug  = re.compile(r'(\d{3})[_\-]?(frontal|mugshot)', re.I)
# SCface ships the SAME surveillance images twice: surveillance_cameras_all/ AND
# surveillance_cameras_distance_N/cam_M/. Dedupe by basename or every probe is counted twice.
seen_surv, dup = set(), 0
# prefer the cropped mugshots (tighter, cleaner) over the originals
all_files.sort(key=lambda p: (0 if 'cropped' in p.lower() else 1,
                              0 if 'surveillance_cameras_all' in p.replace('\\', '/') else 1, p))
for p in all_files:
    b = os.path.basename(p)
    low = p.lower()
    if 'rotation' in low:            # 001_L1.jpg etc - pose variants, not used
        continue
    m = re_mug.search(b)
    if m and ('mug' in low or 'frontal' in b.lower()):
        mug.setdefault(m.group(1), p); continue
    s = re_surv.search(b)
    if s:
        cam = int(s.group(2))
        if not USE_IR and cam > 5: continue    # cam6/7 are IR surveillance; cam8 is the IR mugshot
        if b.lower() in seen_surv: dup += 1; continue
        seen_surv.add(b.lower())
        surv.append((s.group(1), cam, s.group(3), p))
if dup: print(f"[DISCOVER] skipped {dup} duplicate surveillance files (same image in distance/ folders)")
print(f"[DISCOVER] {len(all_files)} images | {len(mug)} mugshots | {len(surv)} surveillance probes")
if not mug or not surv:
    print("[ERROR] could not parse SCface layout. Expected names like 001_frontal.jpg and "
          "001_cam1_1.jpg. Show me `ls` of the folder and I'll fix the regex."); raise SystemExit

# ---------------- enroll: ONE mugshot per subject ----------------
sids = sorted(mug.keys())
rng.shuffle(sids)
enrolled = sorted(sids[:N_ENROLL]); strangers = set(sids[N_ENROLL:])
gc, gn = [], []
for s in enrolled:
    im = cv2.imread(mug[s])
    if im is None: continue
    c, _ = detect_align(im)
    if c is None: print(f"  [warn] no face in mugshot {s}"); continue
    gc.append(c); gn.append(s)
G = embed_batch(gc); gn = np.array(gn)
print(f"[ENROLL] {len(gn)} subjects, ONE mugshot each | {len(strangers)} held out as strangers\n")

# ---------------- probes ----------------
crops, meta, nodet = [], [], 0
for sid, cam, dcode, p in surv:
    im = cv2.imread(p)
    if im is None: continue
    c, px = detect_align(im)
    if c is None: nodet += 1; continue
    crops.append(c); meta.append((sid, cam, dcode, px, sid in strangers))
P = np.concatenate([embed_batch(crops[i:i+256]) for i in range(0, len(crops), 256)], 0)
meta = np.array(meta, dtype=object)
is_str = np.array([m[4] for m in meta], bool)
pxs = np.array([m[3] for m in meta], float)
dcs = np.array([m[2] for m in meta])
cams = np.array([m[1] for m in meta], int)
tsid = np.array([m[0] for m in meta])
print(f"[PROBES] {len(crops)} usable ({nodet} detection failures) | "
      f"{(~is_str).sum()} enrolled, {is_str.sum()} stranger\n")

# verify the distance-code mapping empirically: farther => smaller face
print("[CHECK] distance code vs mean face px (smaller px must = larger metres):")
for dc in sorted(set(dcs)):
    m = dcs == dc
    print(f"   code {dc} -> {DIST_MAP.get(dc,'?')} m   mean {pxs[m].mean():5.1f} px   n={m.sum()}")

# ---------------- scores ----------------
S_raw = P @ G.T
COH = None
if os.path.exists(COHORT_NPZ):
    _c = np.load(COHORT_NPZ, allow_pickle=True)
    COH = _c['cohort'].astype(np.float32); TOPK = int(_c['topk']); ZT = float(_c['z_thresh'])
    c = P @ COH.T
    t = np.partition(c, -TOPK, axis=1)[:, -TOPK:]
    S_z = (S_raw - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)
    print(f"\n[AS-Norm] cohort {COH.shape[0]}, K={TOPK}, DroneFace z_thresh={ZT:.3f}")
else:
    S_z = None
    print(f"\n[AS-Norm] no cohort at {COHORT_NPZ} - only the OLD model will be evaluated")

def evaluate(S, thr):
    top = gn[S.argmax(1)]; best = S.max(1)
    acc = best >= thr
    gen = ~is_str
    tp = int((acc & gen & (top == tsid)).sum())
    fp_name = int((acc & gen & (top != tsid)).sum())
    fp_str = int((acc & is_str).sum())
    fn = int((gen & ~acc).sum())
    prec = tp / max(tp + fp_name + fp_str, 1)
    rec = tp / max(gen.sum(), 1)
    f1 = 2*prec*rec / max(prec+rec, 1e-9)
    far = fp_str / max(is_str.sum(), 1)
    ok = acc & gen & (top == tsid)
    return dict(tp=tp, fp_name=fp_name, fp_str=fp_str, fn=fn, prec=prec, rec=rec, f1=f1,
                far=far, ok=ok, thr=thr)

A = evaluate(S_raw, THRESHOLD)
res = [("A  OLD raw cosine", A)]
if S_z is not None:
    B = evaluate(S_z, ZT); res.append(("B  NEW AS-Norm (DroneFace thr)", B))
    # C: re-tune z on SCface so stranger false-accepts match A exactly
    str_scores = np.sort(S_z[is_str].max(1))
    k = int(np.floor(A['far'] * len(str_scores)))
    thr_c = float(str_scores[len(str_scores)-k-1]) + 1e-6 if k < len(str_scores) else float(str_scores[-1])+1e-6
    C = evaluate(S_z, thr_c); res.append(("C  NEW AS-Norm (matched FAR)", C))

print("\n" + "=" * 84)
print("HEAD-TO-HEAD  (open-set: %d enrolled, %d strangers)" % (len(gn), len(strangers)))
print("=" * 84)
print(f"  {'configuration':<32} {'thr':>7} {'precision':>10} {'recall':>8} {'F1':>7} {'stranger-FA':>12}")
for nm, r in res:
    print(f"  {nm:<32} {r['thr']:>7.3f} {r['prec']:>10.4f} {r['rec']:>8.1%} {r['f1']:>7.4f} {r['far']:>11.2%}")
if S_z is not None:
    print(f"\n  FAIR COMPARISON (A vs C, equal stranger false-accepts):")
    print(f"     recall    {A['rec']:.1%}  ->  {C['rec']:.1%}   ({(C['rec']-A['rec'])*100:+.1f} pts)")
    print(f"     precision {A['prec']:.4f}  ->  {C['prec']:.4f}")
    print(f"     VERDICT: AS-Norm {'WINS' if C['rec'] > A['rec']+0.01 else 'does NOT help'} on SCface")

best_new = res[-1][1] if S_z is not None else A
def breakdown(title, keys, arr):
    print(f"\n  by {title}:")
    print(f"    {'group':>10} {'n':>6} {'OLD':>8} {'NEW':>8} {'delta':>8}")
    for k in keys:
        m = arr == k
        if not m.sum(): continue
        a = A['ok'][m & ~is_str].mean() if (m & ~is_str).sum() else float('nan')
        b = best_new['ok'][m & ~is_str].mean() if (m & ~is_str).sum() else float('nan')
        lab = f"{DIST_MAP.get(k,k)}" if title.startswith('distance') else str(k)
        print(f"    {lab:>10} {int((m & ~is_str).sum()):>6} {a:>7.1%} {b:>7.1%} {b-a:>+8.1%}")

breakdown("distance (m)", sorted(set(dcs)), dcs)
breakdown("camera", sorted(set(cams)), cams)
pxb = np.digitize(pxs, [35, 45, 55, 70, 90, 120])
labs = ['<35', '35-45', '45-55', '55-70', '70-90', '90-120', '120+']
print(f"\n  by face px:")
print(f"    {'band':>10} {'n':>6} {'OLD':>8} {'NEW':>8} {'delta':>8}")
for i, lab in enumerate(labs):
    m = (pxb == i) & ~is_str
    if not m.sum(): continue
    a, b = A['ok'][m].mean(), best_new['ok'][m].mean()
    print(f"    {lab:>10} {int(m.sum()):>6} {a:>7.1%} {b:>7.1%} {b-a:>+8.1%}")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "scface_eval.csv"), 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['subject', 'cam', 'dist_code', 'dist_m', 'px', 'is_stranger',
                'pred_raw', 'sim_raw', 'ok_old', 'z', 'ok_new'])
    topr = gn[S_raw.argmax(1)]
    for i in range(len(meta)):
        w.writerow([tsid[i], cams[i], dcs[i], DIST_MAP.get(dcs[i], ''), round(pxs[i], 1),
                    int(is_str[i]), topr[i], round(float(S_raw[i].max()), 3), int(A['ok'][i]),
                    '' if S_z is None else round(float(S_z[i].max()), 2),
                    int(best_new['ok'][i])])
print(f"\n[SAVED] {os.path.join(OUT_DIR, 'scface_eval.csv')}")

if S_z is not None:
    pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("scface_%H%M%S"))
    os.makedirs(pdir, exist_ok=True)
    ds = sorted(set(dcs), key=lambda k: -DIST_MAP.get(k, 0))
    x = [DIST_MAP.get(k, 0) for k in ds]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, [A['ok'][(dcs == k) & ~is_str].mean() for k in ds], 'o-', color='#7f8c8d', lw=2, label='OLD raw cosine')
    ax.plot(x, [best_new['ok'][(dcs == k) & ~is_str].mean() for k in ds], 'o-', color='#0C7C88', lw=2, label='NEW AS-Norm')
    ax.set_xlabel('distance (m)'); ax.set_ylabel('recall'); ax.set_ylim(0, 1.02)
    ax.set_title('SCface: recall vs distance, equal stranger false-accepts'); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(pdir, 'scface_recall.png'), dpi=150); plt.close(fig)
    print(f"[SAVED] plot -> {pdir}")
