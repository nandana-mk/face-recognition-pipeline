"""
31_droneface_headtohead.py — settle the DroneFace AS-Norm claim with a PROPER cosine sweep.

    python 31_droneface_headtohead.py "C:\\Users\\hticd\\scikit_learn_data\\lfw_home\\lfw_funneled"

WHY THIS EXISTS
  25_score_norm.py reported "AS-Norm +32.7 pts" on DroneFace. That comparison was AS-Norm at its
  optimal threshold versus raw cosine at a FIXED 0.40 - and 0.40 was never shown to be optimal.
  On SCface the identical flaw inflated AS-Norm by ~10 pts: once the cosine threshold was swept,
  raw cosine won at every matched false-accept point (30_sweep_operating_points.py section D).

  This script applies that corrected methodology to DroneFace:
    sweep BOTH decision rules, then compare them at MATCHED numbers of accepted strangers.
  If raw cosine wins here too, the AS-Norm claim is dead everywhere and we ship plain cosine.

PROTOCOL
  GALLERY    11 DroneFace subjects, ONE clean portrait each   (single-reference constraint)
  PROBES     all 1,364 DroneFace crops, 2-17 m, 4 camera heights
  STRANGERS  1,500 LFW faces, degraded the same way, DISJOINT from the AS-Norm cohort
"""
import os, sys, glob
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

BASE   = r"C:\dev\train_val\datasets\raw\DnHFaces\open_data_set"
FACES  = os.path.join(BASE, "photos_all_faces")
PORT   = os.path.join(BASE, "portraits")
TRIO_C = os.path.join(BASE, "trio_cam")
TRIO_G = os.path.join(BASE, "trio_gp")
COHORT_NPZ = r"C:\dev\train_val\models\cohort.npz"
N_STRANGER = 1500
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
SEED = 11
COS_GRID = [0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45, 0.50]
Z_GRID   = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
FA_COUNTS = [0, 1, 3, 8, 15, 30, 75]          # out of N_STRANGER

rng = np.random.default_rng(SEED)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def embed_batch(c):
    f = rec.get_feat(c); f = np.asarray(f, np.float32).reshape(len(c), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
def emb_all(c, bs=256):
    return np.concatenate([embed_batch(c[i:i+bs]) for i in range(0, len(c), bs)], 0)
def detect_largest(im):
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0: return None, 0.0
    j = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
    return kps[j], float(min(bb[j, 2]-bb[j, 0], bb[j, 3]-bb[j, 1]))
def find_frontal(s):
    for fo, pre in [(PORT, f"{s}_"), (TRIO_C, f"{s}_cam_0_porf"), (TRIO_G, f"{s}_gp_0_porf")]:
        if not os.path.isdir(fo): continue
        for f in sorted(os.listdir(fo)):
            if f.lower().startswith(pre) and f.lower().endswith(IMG_EXT):
                return os.path.join(fo, f)
    return None
def degrade(im, kps, w, r):
    k = int(r.integers(30, 81)); s = k / max(w, 1e-6)
    H, W = im.shape[:2]
    sm = cv2.resize(im, (max(16, int(W*s)), max(16, int(H*s))), interpolation=cv2.INTER_AREA)
    if r.random() < 0.75:
        sm = np.clip(sm.astype(np.float32) + r.normal(0, r.uniform(2, 14), sm.shape), 0, 255).astype(np.uint8)
    k2, _ = detect_largest(sm)
    if k2 is None: k2 = kps * s + r.normal(0, 1.0, kps.shape)
    return face_align.norm_crop(sm, k2.astype(np.float32))

# ---------------- gallery + probes ----------------
subs = sorted({f.split('_')[0] for f in os.listdir(FACES) if f.lower().endswith(IMG_EXT)})
names, gc = [], []
for s in subs:
    fp = find_frontal(s)
    if not fp: continue
    im = cv2.imread(fp)
    if im is None: continue
    kps, _ = detect_largest(im)
    gc.append(face_align.norm_crop(im, kps) if kps is not None else cv2.resize(im, (112, 112)))
    names.append(s)
G = emb_all(gc); names = np.array(names)

crops, sids, pxs = [], [], []
for f in os.listdir(FACES):
    if not f.lower().endswith(IMG_EXT): continue
    p = f.split('_')
    if len(p) < 5 or p[1] != 'gp' or p[0] not in names: continue
    im = cv2.imread(os.path.join(FACES, f))
    if im is None: continue
    pxs.append(min(im.shape[:2])); sids.append(p[0])
    crops.append(cv2.resize(im, (112, 112)) if im.shape[:2] != (112, 112) else im)
Pg = emb_all(crops); pxs = np.array(pxs); sids = np.array(sids)
print(f"[GALLERY] {len(names)} subjects, ONE portrait each")
print(f"[PROBES]  {len(crops)} DroneFace crops")

# ---------------- strangers (disjoint from the cohort) ----------------
lfw = sys.argv[1] if len(sys.argv) > 1 else None
if not lfw or not os.path.isdir(lfw):
    print("usage: python 31_droneface_headtohead.py <lfw folder>"); raise SystemExit
files = [p for p in glob.glob(os.path.join(lfw, '**', '*'), recursive=True) if p.lower().endswith(IMG_EXT)]
rng.shuffle(files)
# 26_build_cohort.py consumed the FIRST 2000 usable faces (seed 3); take from the far end so the
# stranger set cannot overlap the cohort.
files = files[::-1]
sc = []
for p in files:
    if len(sc) >= N_STRANGER: break
    im = cv2.imread(p)
    if im is None: continue
    kps, w = detect_largest(im)
    if kps is None or w < 50: continue
    sc.append(degrade(im, kps, w, rng))
Ps = emb_all(sc)
print(f"[STRANGERS] {len(sc)} LFW faces, degraded\n")

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
    tp = int((acc & (top == sids)).sum()); wrong = int((acc & (top != sids)).sum())
    fa = int((SS.max(1) >= thr).sum())
    prec = tp / max(tp + wrong + fa, 1); rec = tp / len(SG)
    return prec, rec, 2*prec*rec/max(prec+rec, 1e-9), fa, wrong, acc & (top == sids)

def thr_at_fa(SS, n):
    sm = np.sort(SS.max(1))[::-1]
    return float(sm[n]) + 1e-6 if n < len(sm) else float(sm[-1]) - 1e-6

print("=" * 88)
print("A. RAW COSINE — threshold sweep")
print("=" * 88)
print(f"  {'thr':>7} {'precision':>10} {'recall':>8} {'F1':>8} {'#FA':>6} {'FA%':>8} {'wrong-name':>11}")
for t in COS_GRID:
    p, r, f1, fa, wr, _ = metrics(Sg, Ss, t)
    print(f"  {t:>7.3f} {p:>10.4f} {r:>8.1%} {f1:>8.4f} {fa:>6d} {fa/len(Ss):>7.2%} {wr:>11d}")

print("\n" + "=" * 88)
print("B. AS-NORM — z-threshold sweep")
print("=" * 88)
print(f"  {'z':>7} {'precision':>10} {'recall':>8} {'F1':>8} {'#FA':>6} {'FA%':>8} {'wrong-name':>11}")
for t in Z_GRID:
    p, r, f1, fa, wr, _ = metrics(Zg, Zs, t)
    print(f"  {t:>7.2f} {p:>10.4f} {r:>8.1%} {f1:>8.4f} {fa:>6d} {fa/len(Ss):>7.2%} {wr:>11d}")

print("\n" + "=" * 88)
print("C. HEAD-TO-HEAD at MATCHED accepted strangers   <-- THE DECISIVE TEST")
print("=" * 88)
print(f"  {'#FA':>4} {'FA%':>7} | {'COS thr':>8} {'recall':>8} {'prec':>7} | "
      f"{'Z thr':>8} {'recall':>8} {'prec':>7} | {'recall Δ':>9}")
wins = {'cos': 0, 'z': 0}
best_pair = None
for n in FA_COUNTS:
    if n >= len(Ss): continue
    tc = thr_at_fa(Ss, n); pc, rc, f1c, _, _, okc = metrics(Sg, Ss, tc)
    tz = thr_at_fa(Zs, n); pz, rz, f1z, _, _, okz = metrics(Zg, Zs, tz)
    wins['cos' if rc >= rz else 'z'] += 1
    if n == 15: best_pair = (tc, okc, tz, okz)
    print(f"  {n:>4} {n/len(Ss):>6.2%} | {tc:>8.3f} {rc:>8.1%} {pc:>7.4f} | "
          f"{tz:>8.3f} {rz:>8.1%} {pz:>7.4f} | {(rz-rc)*100:>+8.1f}")

print(f"\n  matched-FA points won:  raw cosine {wins['cos']}   AS-Norm {wins['z']}")

if best_pair:
    tc, okc, tz, okz = best_pair
    print(f"\n  per-pixel-band recall at 1% FA  (cos thr {tc:.3f} vs z thr {tz:.3f}):")
    print(f"    {'band':>9} {'n':>5} {'COS':>8} {'AS-Norm':>9} {'delta':>8}")
    for lo, hi, lab in [(0,35,'<35'), (35,45,'35-45'), (45,55,'45-55'), (55,70,'55-70'),
                        (70,90,'70-90'), (90,120,'90-120'), (120,10**6,'120+')]:
        m = (pxs >= lo) & (pxs < hi)
        if not m.sum(): continue
        print(f"    {lab:>9} {int(m.sum()):>5} {okc[m].mean():>7.1%} {okz[m].mean():>8.1%} "
              f"{(okz[m].mean()-okc[m].mean())*100:>+7.1f}")

print("\n" + "=" * 88)
print("VERDICT")
print("=" * 88)
_, r40, _, fa40, _, _ = metrics(Sg, Ss, 0.40)
tbest, rbest = None, -1
for t in np.arange(0.15, 0.55, 0.005):
    p, r, f1, fa, wr, _ = metrics(Sg, Ss, float(t))
    if fa <= fa40 and r > rbest: tbest, rbest = float(t), r
print(f"  raw cosine @ 0.40 (the old baseline): recall {r40:.1%}, {fa40} strangers accepted")
print(f"  raw cosine best at the SAME FA count: thr {tbest:.3f}, recall {rbest:.1%} "
      f"({(rbest-r40)*100:+.1f} pts from tuning the threshold ALONE)")
if wins['cos'] > wins['z']:
    print("\n  >>> raw cosine wins on DroneFace too. The '+32.7 pts' was a threshold artifact.")
    print("      Ship plain cosine with a calibrated threshold; drop AS-Norm and the cohort.")
else:
    print("\n  >>> AS-Norm still wins here. DroneFace and SCface disagree - report both and")
    print("      decide on YOUR data, which is what 28_compare_labeled.py measures.")
