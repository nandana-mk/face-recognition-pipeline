"""
26_build_cohort.py — build the deployable AS-Norm cohort, and calibrate it.

    python 26_build_cohort.py "C:\\Users\\hticd\\scikit_learn_data\\lfw_home\\lfw_funneled"

Produces ONE artifact that ships next to gallery.npz and contains everything the runtime needs:
    models/cohort.npz
        cohort[M,512]   background embeddings (faces the system should never match)
        topk            how many of the most-confusable cohort scores to normalise against
        z_thresh        the accept threshold, in standard deviations
        domain          'clean' or 'degraded' - which cohort variant won the sweep

It also SWEEPS the two design choices I flagged as untuned:
    cohort domain : clean vs degraded  (we normalise probe-vs-CLEAN-gallery scores, so a clean
                    cohort is arguably the principled choice - the first run used degraded)
    top-K         : 50 / 100 / 200 / 400

Every configuration is compared at MATCHED stranger false-accept rate on held-out LFW faces,
so "more recall" can never just be a looser threshold.

Calibration uses DroneFace (11 identities, single-portrait enrollment, automatic ground truth).
The z-threshold transfers better than a cosine threshold because it is scale-free, but it should
still be re-checked on your own watchlist before it is trusted operationally.
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
OUT    = r"C:\dev\train_val\models\cohort.npz"
N_COHORT, N_IMPOSTOR = 2000, 1500
KS = [50, 100, 200, 400]
FAR_TARGETS = [0.0007, 0.002, 0.005]      # 0.07%, 0.2%, 0.5% stranger false-accept budgets
FAR_PICK = 0.002                          # config chosen at this budget
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
SEED = 3

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
            if f.lower().startswith(pre) and f.lower().endswith(IMG_EXT): return os.path.join(fo, f)
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

# ---------------- gallery + probes (DroneFace) ----------------
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
G = emb_all(gc)
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
print(f"[INFO] gallery {len(names)} (one photo each), probes {len(crops)}\n")

# ---------------- LFW: cohort (clean + degraded) and disjoint impostors ----------------
lfw = sys.argv[1] if len(sys.argv) > 1 else None
if not lfw or not os.path.isdir(lfw):
    print("usage: python 26_build_cohort.py <lfw folder>"); raise SystemExit
files = [p for p in glob.glob(os.path.join(lfw, '**', '*'), recursive=True) if p.lower().endswith(IMG_EXT)]
rng.shuffle(files)
coh_cl, coh_dg, imp_dg = [], [], []
for p in files:
    if len(coh_cl) >= N_COHORT and len(imp_dg) >= N_IMPOSTOR: break
    im = cv2.imread(p)
    if im is None: continue
    kps, w = detect_largest(im)
    if kps is None or w < 50: continue
    if len(coh_cl) < N_COHORT:
        coh_cl.append(face_align.norm_crop(im, kps)); coh_dg.append(degrade(im, kps, w, rng))
    else:
        imp_dg.append(degrade(im, kps, w, rng))
C_clean, C_deg, I = emb_all(coh_cl), emb_all(coh_dg), emb_all(imp_dg)
print(f"[INFO] cohort {C_clean.shape[0]} (clean+degraded), impostors {I.shape[0]} (disjoint)\n")

# ---------------- sweep ----------------
def zscores(E, C, k):
    s = E @ G.T; c = E @ C.T
    t = np.partition(c, -k, axis=1)[:, -k:]
    return (s - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)

def recall_at_far(Zp, Zi, far):
    """threshold set so stranger false-accepts <= far, then measure recall"""
    thr = float(np.quantile(Zi.max(1), 1 - far))
    ok = (np.array(names)[Zp.argmax(1)] == sids) & (Zp.max(1) >= thr)
    return ok.mean(), thr, float((Zi.max(1) >= thr).mean()), ok

raw_ok = (np.array(names)[(Pr @ G.T).argmax(1)] == sids) & ((Pr @ G.T).max(1) >= 0.40)
raw_far = float(((I @ G.T).max(1) >= 0.40).mean())
print(f"[BASELINE] raw cosine @0.40  recall {raw_ok.mean():.1%}  stranger-FA {raw_far:.2%}\n")

print("=" * 76)
print("SWEEP - recall at matched stranger false-accept rate")
print("=" * 76)
hdr = "  " + f"{'cohort':>9} {'K':>5} " + " ".join(f"{'FA'+f'{t:.2%}':>12}" for t in FAR_TARGETS)
print(hdr)
best = None
for dom, C in [('clean', C_clean), ('degraded', C_deg)]:
    for k in KS:
        if k > C.shape[0]: continue
        Zp, Zi = zscores(Pr, C, k), zscores(I, C, k)
        row, cells = [], []
        for far in FAR_TARGETS:
            r, thr, afar, ok = recall_at_far(Zp, Zi, far)
            cells.append(f"{r:>11.1%}")
            if abs(far - FAR_PICK) < 1e-12:
                if best is None or r > best[0]: best = (r, dom, k, thr, ok, C)
        print(f"  {dom:>9} {k:>5} " + " ".join(cells))

r_best, dom_best, k_best, thr_best, ok_best, C_best = best
print(f"\n  BEST @ FA<={FAR_PICK:.2%}:  cohort={dom_best}  K={k_best}  "
      f"z_thresh={thr_best:.3f}  recall={r_best:.1%}")
print(f"  vs raw cosine @0.40: {raw_ok.mean():.1%}  ->  {r_best:.1%}  "
      f"({(r_best-raw_ok.mean())*100:+.1f} pts)")

print(f"\n  {'px band':>9} {'n':>5} {'raw':>8} {'AS-Norm':>9} {'delta':>8}")
for lo, hi, lab in [(0,35,'<35'), (35,45,'35-45'), (45,55,'45-55'), (55,70,'55-70'),
                    (70,90,'70-90'), (90,120,'90-120'), (120,10**6,'120+')]:
    m = (pxs >= lo) & (pxs < hi)
    if not m.sum(): continue
    a, b = raw_ok[m].mean(), ok_best[m].mean()
    print(f"  {lab:>9} {int(m.sum()):>5} {a:>7.1%} {b:>8.1%} {b-a:>+8.1%}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez_compressed(OUT, cohort=C_best.astype(np.float32), topk=np.int32(k_best),
                    z_thresh=np.float32(thr_best), domain=np.str_(dom_best))
print(f"\n[SAVED] {OUT}")
print(f"  cohort {C_best.shape}  topk={k_best}  z_thresh={thr_best:.3f}  domain={dom_best}")
print("Next: python 21_run_folder.py   (it will pick this up automatically)")
