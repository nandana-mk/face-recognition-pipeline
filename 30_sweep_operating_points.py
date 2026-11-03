"""
30_sweep_operating_points.py — pick the operating point with evidence, single-reference enrollment.

    python 30_sweep_operating_points.py            # uses the folders 29_calibrate_scface.py built

Sweeps BOTH decision rules over the same SCface data (100 identities, ONE mugshot each):
    raw cosine  thresholds 0.35 .. 0.60
    AS-Norm     at stranger false-accept budgets 0% .. 2%
and reports precision / recall / F1 / stranger-FA / wrong-name accepts for every point.

The question it answers: "is a higher cosine threshold (0.45+) better?" - and, if precision is
already perfect at 0.40, what raising it actually costs you.
"""
import os, glob
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

WORK       = r"C:\dev\train_val\calib_scface"
COHORT_NPZ = r"C:\dev\train_val\models\cohort.npz"
IMG_EXT    = ('.jpg', '.jpeg', '.png', '.bmp')
COS_GRID   = [0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60]
FAR_GRID   = [0.0, 0.002, 0.005, 0.01, 0.02]
# explicit z-grid requested for inspection
Z_GRID     = [4.0, 4.5, 4.7, 5.0, 5.5, 5.7, 5.9, 6.0, 6.2, 6.5, 6.75, 7.0, 7.25, 7.5, 8.0, 8.5]
# head-to-head is done at MATCHED numbers of accepted strangers (the only fair comparison)
FA_COUNTS  = [0, 1, 2, 3, 5, 9, 14, 23]

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def embed_batch(c):
    f = rec.get_feat(c); f = np.asarray(f, np.float32).reshape(len(c), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)

def align_largest(img):
    """same fallback chain as the module: detect -> detect padded -> accept as a face crop"""
    bb, kps = det.detect(img, max_num=0, metric='default')
    if bb is not None and len(bb):
        j = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        return face_align.norm_crop(img, kps[j])
    h, w = img.shape[:2]; pad = int(0.4*max(h, w))
    p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    bb, kps = det.detect(p, max_num=0, metric='default')
    if bb is not None and len(bb):
        j = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        return face_align.norm_crop(p, kps[j])
    if h and w and 0.6 <= w/h <= 1.8 and min(h, w) >= 60:
        return cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    return None

def load_dir(d, recursive):
    pat = os.path.join(d, '**', '*') if recursive else os.path.join(d, '*')
    crops = []
    for p in sorted(glob.glob(pat, recursive=recursive)):
        if not p.lower().endswith(IMG_EXT): continue
        im = cv2.imread(p)
        if im is None: continue
        c = align_largest(im)
        if c is not None: crops.append(c)
    return crops

# ---------------- gallery (ONE reference per identity) ----------------
gdir = os.path.join(WORK, 'gallery')
if not os.path.isdir(gdir):
    print(f"[ERROR] {gdir} not found - run 29_calibrate_scface.py first"); raise SystemExit
names, gc = [], []
for person in sorted(os.listdir(gdir)):
    pd = os.path.join(gdir, person)
    if not os.path.isdir(pd): continue
    cs = load_dir(pd, False)
    if cs: names.append(person); gc.append(cs[0])
G = embed_batch(gc); names = np.array(names)
print(f"[GALLERY] {len(names)} identities, ONE reference each")

# ---------------- probes ----------------
gen_vecs, gen_true = [], []
for person in sorted(os.listdir(os.path.join(WORK, 'genuine'))):
    pd = os.path.join(WORK, 'genuine', person)
    if not os.path.isdir(pd) or person not in names: continue
    cs = load_dir(pd, True)
    if not cs: continue
    gen_vecs.append(embed_batch(cs)); gen_true += [person]*len(cs)
Pg = np.concatenate(gen_vecs, 0); gen_true = np.array(gen_true)
str_crops = load_dir(os.path.join(WORK, 'strangers'), True)
Ps = embed_batch(str_crops)
print(f"[PROBES] genuine {len(Pg)}, strangers {len(Ps)}\n")

# ---------------- scores ----------------
Sg, Ss = Pg @ G.T, Ps @ G.T
d = np.load(COHORT_NPZ, allow_pickle=True)
COH = d['cohort'].astype(np.float32); K = int(d['topk'])
def zs(S, P):
    c = P @ COH.T
    t = np.partition(c, -K, axis=1)[:, -K:]
    return (S - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)
Zg, Zs = zs(Sg, Pg), zs(Ss, Ps)

def metrics(SG, SS, thr):
    top = names[SG.argmax(1)]; acc = SG.max(1) >= thr
    tp = int((acc & (top == gen_true)).sum())
    wrong = int((acc & (top != gen_true)).sum())
    fa = int((SS.max(1) >= thr).sum())
    prec = tp / max(tp + wrong + fa, 1); rec = tp / len(SG)
    return prec, rec, 2*prec*rec/max(prec+rec, 1e-9), fa/len(SS), wrong

print("=" * 82)
print("A. RAW COSINE (the pre-AS-Norm module) — single reference")
print("=" * 82)
print(f"  {'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>8} {'stranger-FA':>12} {'wrong-name':>11}")
best_cos = None
for t in COS_GRID:
    p, r, f1, far, wr = metrics(Sg, Ss, t)
    mark = ""
    if best_cos is None or f1 > best_cos[0]: best_cos = (f1, t, p, r, far); mark = "  <- best F1"
    print(f"  {t:>10.3f} {p:>10.4f} {r:>8.1%} {f1:>8.4f} {far:>11.2%} {wr:>11d}{mark}")

print("\n" + "=" * 82)
print("B. AS-NORM — single reference, threshold set by false-accept budget")
print("=" * 82)
print(f"  {'FA budget':>10} {'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>8} "
      f"{'actual FA':>10} {'wrong-name':>11}")
sm = np.sort(Zs.max(1))[::-1]
best_z = None
for far in FAR_GRID:
    allowed = int(np.floor(far*len(sm)))
    thr = float(sm[allowed]) + 1e-6 if allowed < len(sm) else float(sm[-1]) - 1e-6
    p, r, f1, afar, wr = metrics(Zg, Zs, thr)
    mark = ""
    if best_z is None or f1 > best_z[0]: best_z = (f1, thr, p, r, afar); mark = "  <- best F1"
    print(f"  {far:>9.1%} {thr:>10.4f} {p:>10.4f} {r:>8.1%} {f1:>8.4f} {afar:>9.2%} {wr:>11d}{mark}")

print("\n" + "=" * 82)
print("C. AS-NORM — explicit z-threshold grid")
print("=" * 82)
print(f"  {'z_thresh':>9} {'precision':>10} {'recall':>8} {'F1':>8} {'stranger-FA':>12} "
      f"{'wrong-name':>11} {'#FA':>5}")
best_grid = None
for t in Z_GRID:
    p, r, f1, far, wr = metrics(Zg, Zs, float(t))
    n_fa = int((Zs.max(1) >= t).sum())
    mark = ""
    if best_grid is None or f1 > best_grid[0]:
        best_grid = (f1, float(t), p, r, far); mark = "  <- best F1"
    print(f"  {t:>9.2f} {p:>10.4f} {r:>8.1%} {f1:>8.4f} {far:>11.2%} {wr:>11d} {n_fa:>5d}{mark}")

# ---------------- D. proper head-to-head at MATCHED accepted-stranger counts ----------------
def thr_at_fa(S_str, n_allowed):
    sm = np.sort(S_str.max(1))[::-1]
    if n_allowed >= len(sm):
        return float(sm[-1]) - 1e-6
    return float(sm[n_allowed]) + 1e-6

print("\n" + "=" * 82)
print("D. HEAD-TO-HEAD at MATCHED stranger false-accepts  (the fair validation)")
print("=" * 82)
print(f"  strangers accepted out of {len(Zs)}  |  both modules forced to the same precision cost")
print(f"\n  {'#FA':>4} {'FA%':>7} | {'COS thr':>8} {'prec':>7} {'recall':>8} {'F1':>7} | "
      f"{'Z thr':>8} {'prec':>7} {'recall':>8} {'F1':>7} | {'recall Δ':>9}")
for n in FA_COUNTS:
    if n >= len(Zs): continue
    tc_ = thr_at_fa(Ss, n); pc_, rc_, f1c_, farc_, wc_ = metrics(Sg, Ss, tc_)
    tz_ = thr_at_fa(Zs, n); pz_, rz_, f1z_, farz_, wz_ = metrics(Zg, Zs, tz_)
    print(f"  {n:>4} {n/len(Zs):>6.2%} | {tc_:>8.3f} {pc_:>7.4f} {rc_:>8.1%} {f1c_:>7.4f} | "
          f"{tz_:>8.3f} {pz_:>7.4f} {rz_:>8.1%} {f1z_:>7.4f} | {(rz_-rc_)*100:>+8.1f}")

print("\n" + "=" * 82)
print("VERDICT")
print("=" * 82)
f1c, tc, pc, rc, farc = best_cos
f1z, tz, pz, rz, farz = best_z
print(f"  best RAW COSINE : thr {tc:.3f}  precision {pc:.4f}  recall {rc:.1%}  F1 {f1c:.4f}  FA {farc:.2%}")
print(f"  best AS-NORM    : thr {tz:.3f}  precision {pz:.4f}  recall {rz:.1%}  F1 {f1z:.4f}  FA {farz:.2%}")
f1g, tg, pg_, rg_, farg = best_grid
print(f"  best on GRID    : z {tg:.1f}      precision {pg_:.4f}  recall {rg_:.1%}  F1 {f1g:.4f}  FA {farg:.2%}")
p40 = metrics(Sg, Ss, 0.40); p45 = metrics(Sg, Ss, 0.45)
print(f"\n  Cost of raising cosine 0.40 -> 0.45:")
print(f"     recall    {p40[1]:.1%} -> {p45[1]:.1%}  ({(p45[1]-p40[1])*100:+.1f} pts)")
print(f"     precision {p40[0]:.4f} -> {p45[0]:.4f}   stranger-FA {p40[3]:.2%} -> {p45[3]:.2%}")
if p40[0] >= 0.9999 and p40[3] <= 0.0:
    print("     -> precision is ALREADY perfect at 0.40; raising it buys nothing and costs recall.")
