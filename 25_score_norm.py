"""
25_score_norm.py — fix the REAL failure: correct identity, score below threshold.

    python 25_score_norm.py "C:\\Users\\hticd\\scikit_learn_data\\lfw_home\\lfw_funneled"

WHY THIS AND NOT THE ADAPTER
  The adapter gave 0 gain because degradation preserves RANKING (top-1 held at 93.8% while cosine
  fell to 0.509). The embeddings point the right way; their ABSOLUTE similarity collapses.
  Evidence from our own data:
     DroneFace : margin>0 (correct nearest identity) on 65.4% of crops, but recall only 24.4%
                 -> ~41% of crops have the RIGHT answer and are rejected by the 0.40 threshold
     DJI_001   : 4 of 7 missed people had the CORRECT top guess (0.39/0.39/0.38/0.23)
  That is a score-calibration problem, not a representation problem.

AS-NORM (adaptive score normalisation)
  A degraded probe scores low against EVERYTHING, including unrelated faces. So measure how low,
  and subtract it:
        s_norm(i) = ( s(i) - mean(top-K cohort scores) ) / std(top-K cohort scores)
  where the cohort is a set of background identities the probe is NOT supposed to match.
  No training. Works with a single reference photo per person.

HONEST COMPARISON
  Comparing recall at "the same threshold" is meaningless - normalised scores live on a different
  scale. So we pick the AS-Norm threshold that gives the SAME stranger false-accept rate as the
  raw 0.40 operating point, then compare recall. Equal precision, measured recall.
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
THRESHOLD = 0.40
N_COHORT, N_IMPOSTOR, TOPK = 2000, 1500, 200
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
SEED = 3

rng = np.random.default_rng(SEED)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def embed_batch(crops):
    f = rec.get_feat(crops)
    f = np.asarray(f, np.float32).reshape(len(crops), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)

def emb_all(crops, bs=256):
    return np.concatenate([embed_batch(crops[i:i+bs]) for i in range(0, len(crops), bs)], 0)

def detect_largest(im):
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0: return None, 0.0
    j = int(np.argmax((bb[:, 2]-bb[:, 0]) * (bb[:, 3]-bb[:, 1])))
    return kps[j], float(min(bb[j, 2]-bb[j, 0], bb[j, 3]-bb[j, 1]))

def find_frontal(s):
    for folder, pre in [(PORT, f"{s}_"), (TRIO_C, f"{s}_cam_0_porf"), (TRIO_G, f"{s}_gp_0_porf")]:
        if not os.path.isdir(folder): continue
        for f in sorted(os.listdir(folder)):
            if f.lower().startswith(pre) and f.lower().endswith(IMG_EXT):
                return os.path.join(folder, f)
    return None

def degrade_like_drone(im, kps, w, r):
    """same realistic chain as training: shrink whole image, corrupt, re-detect, align"""
    k = int(r.integers(30, 81)); s = k / max(w, 1e-6)
    H, W = im.shape[:2]
    small = cv2.resize(im, (max(16, int(W*s)), max(16, int(H*s))), interpolation=cv2.INTER_AREA)
    if r.random() < 0.75:
        small = np.clip(small.astype(np.float32) + r.normal(0, r.uniform(2, 14), small.shape),
                        0, 255).astype(np.uint8)
    kps_s, _ = detect_largest(small)
    if kps_s is None: kps_s = kps * s + r.normal(0, 1.0, kps.shape)
    return face_align.norm_crop(small, kps_s.astype(np.float32))

# ---------------- gallery: ONE portrait per subject ----------------
subjects = sorted({f.split('_')[0] for f in os.listdir(FACES) if f.lower().endswith(IMG_EXT)})
names, gc = [], []
for s in subjects:
    fp = find_frontal(s)
    if not fp: continue
    im = cv2.imread(fp)
    if im is None: continue
    kps, _ = detect_largest(im)
    gc.append(face_align.norm_crop(im, kps) if kps is not None else cv2.resize(im, (112, 112)))
    names.append(s)
G = emb_all(gc)
print(f"[INFO] gallery: {len(names)} subjects, ONE reference photo each\n")

# ---------------- probes: DroneFace crops ----------------
crops, sids, pxs = [], [], []
for f in os.listdir(FACES):
    if not f.lower().endswith(IMG_EXT): continue
    p = f.split('_')
    if len(p) < 5 or p[1] != 'gp' or p[0] not in names: continue
    im = cv2.imread(os.path.join(FACES, f))
    if im is None: continue
    pxs.append(min(im.shape[:2])); sids.append(p[0])
    crops.append(cv2.resize(im, (112, 112)) if im.shape[:2] != (112, 112) else im)
Pr = emb_all(crops); pxs = np.array(pxs); sids = np.array(sids)
print(f"[INFO] {len(crops)} DroneFace probes")

# ---------------- cohort + impostors from LFW (disjoint sets) ----------------
lfw = sys.argv[1] if len(sys.argv) > 1 else None
if not lfw or not os.path.isdir(lfw):
    print("usage: python 25_score_norm.py <lfw folder>"); raise SystemExit
files = [p for p in glob.glob(os.path.join(lfw, '**', '*'), recursive=True) if p.lower().endswith(IMG_EXT)]
rng.shuffle(files)
need = N_COHORT + N_IMPOSTOR
coh_c, imp_c = [], []
for p in files:
    if len(coh_c) + len(imp_c) >= need: break
    im = cv2.imread(p)
    if im is None: continue
    kps, w = detect_largest(im)
    if kps is None or w < 50: continue
    d = degrade_like_drone(im, kps, w, rng)          # cohort/impostors must match probe domain
    (coh_c if len(coh_c) < N_COHORT else imp_c).append(d)
C = emb_all(coh_c); I = emb_all(imp_c)
print(f"[INFO] cohort {C.shape[0]}  impostors {I.shape[0]}  (disjoint LFW faces, degraded)\n")

# ---------------- scoring ----------------
def asnorm(E, G, C, k=TOPK):
    """probe-side adaptive score normalisation"""
    s = E @ G.T                                   # (N, K) raw scores vs gallery
    c = E @ C.T                                   # (N, M) scores vs cohort
    t = np.sort(c, axis=1)[:, -k:]                # top-k cohort scores per probe
    mu = t.mean(1, keepdims=True); sd = t.std(1, keepdims=True) + 1e-6
    return s, (s - mu) / sd

S_raw, S_norm = asnorm(Pr, G, C)
I_raw, I_norm = asnorm(I, G, C)

def stats(S, thr):
    top = S.argmax(1); best = S.max(1)
    ok = (np.array(names)[top] == sids) & (best >= thr)
    return ok, best

def far(SI, thr):
    return float((SI.max(1) >= thr).mean())

# operating point: raw @ 0.40 -> measure its stranger false-accept rate
far_raw = far(I_raw, THRESHOLD)
ok_raw, _ = stats(S_raw, THRESHOLD)

# find the AS-Norm threshold with the SAME false-accept rate (equal precision comparison)
cand = np.quantile(I_norm.max(1), np.linspace(0.90, 1.0, 2001))
thr_norm = None
for t in cand:
    if far(I_norm, t) <= far_raw: thr_norm = t; break
if thr_norm is None: thr_norm = cand[-1]
ok_norm, _ = stats(S_norm, thr_norm)

print("=" * 72)
print("EQUAL-PRECISION COMPARISON  (single reference photo per person)")
print("=" * 72)
print(f"  raw      threshold {THRESHOLD:.3f}   stranger false-accept {far_raw:.2%}")
print(f"  AS-Norm  threshold {thr_norm:.3f}   stranger false-accept {far(I_norm, thr_norm):.2%}")
print(f"\n  OVERALL RECALL   raw {ok_raw.mean():.1%}  ->  AS-Norm {ok_norm.mean():.1%}   "
      f"({(ok_norm.mean()-ok_raw.mean())*100:+.1f} pts)")

print(f"\n  {'px band':>9} {'n':>5} {'raw':>8} {'AS-Norm':>9} {'delta':>8}")
for lo, hi, lab in [(0,35,'<35'), (35,45,'35-45'), (45,55,'45-55'), (55,70,'55-70'),
                    (70,90,'70-90'), (90,120,'90-120'), (120,10**6,'120+')]:
    m = (pxs >= lo) & (pxs < hi)
    if not m.sum(): continue
    a, b = ok_raw[m].mean(), ok_norm[m].mean()
    print(f"  {lab:>9} {int(m.sum()):>5} {a:>7.1%} {b:>8.1%} {b-a:>+8.1%}")

# how much of the recoverable pool did we get?
top_ok = np.array(names)[S_raw.argmax(1)] == sids
pool = top_ok & ~ok_raw
print(f"\n  RECOVERABLE POOL (correct top-1 but rejected by raw threshold): "
      f"{pool.sum()} crops = {pool.mean():.1%}")
print(f"  of those, AS-Norm now accepts: {(pool & ok_norm).sum()} ({(pool & ok_norm).sum()/max(pool.sum(),1):.1%})")

print("\n" + "=" * 72)
if ok_norm.mean() > ok_raw.mean() + 0.02:
    print("  >>> AS-Norm WINS at equal stranger-rejection. Deploy it: compute cohort scores once")
    print("      (a fixed set of background embeddings ships with the gallery), normalise every")
    print("      probe score, and threshold on the normalised value.")
else:
    print("  >>> No meaningful gain. The score depression is not probe-specific enough for AS-Norm;")
    print("      fall back to the face-size-banded thresholds from 15_calibrate_decision.py.")
