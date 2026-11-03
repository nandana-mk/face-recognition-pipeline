"""
33_distance_envelope.py — how far can we go at high precision AND recall?

    python 33_distance_envelope.py "C:\\Users\\hticd\\scikit_learn_data\\lfw_home\\lfw_funneled"

Answers three questions, at the DEPLOYED threshold (and 0.40 for comparison):
  1. recall and identity-confusion vs DISTANCE and vs CAMERA HEIGHT (DroneFace, 2-17 m)
  2. the camera-agnostic limit: recall vs FACE PIXELS  (this is the real physical law)
  3. the range spec: how many face-pixels-per-metre (k) your optics need to reach a target range

WHY POINT 3 MATTERS
    Distance in metres is NOT a property of the recogniser. Face pixels are. The relationship is
    a pure inverse law, face_px = k / distance, where k is a constant of YOUR camera+lens.
    DroneFace's camera measured k = 243. So its 66 px (50% recall) limit falls at 3.7 m.
    A camera with a larger k reaches the same pixel count further away. This script converts the
    measured pixel limits into the k your drone needs for a target range - a hardware spec.

Precision is measured open-set: DroneFace genuine probes + LFW strangers (never enrolled), so
the threshold is held to a real precision budget rather than assumed.
"""
import os, sys, glob
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, datetime
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
THRESHOLDS = [0.284, 0.40]        # deployed vs old
N_STRANGER = 4000
RECALL_TARGETS = [0.80, 0.50]
TARGET_RANGES = [5.0, 8.0, 10.0]  # metres we would LIKE to reach
K_DRONEFACE = 243.0               # measured: face_px = 243 / distance on DroneFace's camera
K_YOUR_DRONE = None               # set once you calibrate your DJI (see the note at the end)
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
SEED = 5

rng = np.random.default_rng(SEED)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def emb(c):
    f = rec.get_feat(c); f = np.asarray(f, np.float32).reshape(len(c), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
def emb_all(c, bs=256):
    return np.concatenate([emb(c[i:i+bs]) for i in range(0, len(c), bs)], 0)
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
    k = int(r.integers(25, 80)); s = k / max(w, 1e-6)
    H, W = im.shape[:2]
    sm = cv2.resize(im, (max(16, int(W*s)), max(16, int(H*s))), interpolation=cv2.INTER_AREA)
    if r.random() < 0.8:
        sm = np.clip(sm.astype(np.float32) + r.normal(0, r.uniform(3, 14), sm.shape), 0, 255).astype(np.uint8)
    k2, _ = detect_largest(sm)
    if k2 is None: k2 = kps*s + r.normal(0, 1.0, kps.shape)
    return face_align.norm_crop(sm, k2.astype(np.float32))

# ---------------- gallery + probes ----------------
HEIGHT = {'0': 1.5, '3': 3.0, '4': 4.0, '5': 5.0}
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

crops, sid, dist, hgt, px = [], [], [], [], []
for f in os.listdir(FACES):
    if not f.lower().endswith(IMG_EXT): continue
    p = f.split('_')
    if len(p) < 5 or p[1] != 'gp' or p[0] not in names: continue
    im = cv2.imread(os.path.join(FACES, f))
    if im is None: continue
    crops.append(cv2.resize(im, (112, 112)) if im.shape[:2] != (112, 112) else im)
    sid.append(p[0]); dist.append(17 - int(p[4].split('.')[0])/2.0)
    hgt.append(HEIGHT.get(p[2], 0)); px.append(min(im.shape[:2]))
P = emb_all(crops)
sid = np.array(sid); dist = np.array(dist); hgt = np.array(hgt); px = np.array(px, float)
print(f"[DroneFace] {len(names)} enrolled (one portrait each), {len(crops)} probes\n")

# ---------------- strangers ----------------
lfw = sys.argv[1] if len(sys.argv) > 1 else None
S_str = np.zeros((0, len(names)), np.float32)
if lfw and os.path.isdir(lfw):
    fl = [p for p in glob.glob(os.path.join(lfw, '**', '*'), recursive=True) if p.lower().endswith(IMG_EXT)]
    rng.shuffle(fl); sc = []
    for p in fl:
        if len(sc) >= N_STRANGER: break
        im = cv2.imread(p)
        if im is None: continue
        kps, w = detect_largest(im)
        if kps is None or w < 50: continue
        sc.append(degrade(im, kps, w, rng))
    S_str = emb_all(sc) @ G.T
    print(f"[strangers] {S_str.shape[0]} degraded LFW faces\n")

S = P @ G.T
top = names[S.argmax(1)]; best = S.max(1)
correct = (top == sid)

def report(thr):
    acc = best >= thr
    ok = acc & correct
    wrong = int(np.sum(acc & ~correct))
    fa = int(np.sum(S_str.max(1) >= thr)) if S_str.shape[0] else 0
    prec = int(ok.sum()) / max(int(ok.sum()) + wrong + fa, 1)
    print("=" * 84)
    print(f"THRESHOLD {thr:.3f}   overall precision {prec:.4f}   "
          f"(wrong-name {wrong}, stranger-accepts {fa}/{S_str.shape[0]})")
    print("=" * 84)

    print(f"  {'dist(m)':>8} {'n':>5} {'mean px':>8} {'recall':>8} {'wrong':>6}")
    dd = sorted(set(dist))
    env = {}
    for d in dd:
        m = dist == d
        r = ok[m].mean(); w = int(np.sum(acc[m] & ~correct[m]))
        print(f"  {d:>8.1f} {int(m.sum()):>5} {px[m].mean():>8.0f} {r:>7.1%} {w:>6d}")
    for tgt in RECALL_TARGETS:
        good = [d for d in dd if ok[dist == d].mean() >= tgt]
        env[tgt] = max(good) if good else None
        s = f"{max(good):.1f} m" if good else "never reached"
        print(f"  --> recall >= {tgt:.0%} out to: {s}")

    print(f"\n  by camera height (recall at this threshold):")
    for h in sorted(set(hgt)):
        m = hgt == h
        good = [d for d in dd if ok[m & (dist == d)].mean() >= 0.80] if m.sum() else []
        s = f"{max(good):.1f} m" if good else "never"
        print(f"    {h:>4.1f} m camera: overall {ok[m].mean():>6.1%}   recall>=80% out to {s}")

    print(f"\n  by face pixels (the camera-agnostic law):")
    pxl = None
    for lo, hi, lab in [(0,35,'<35'), (35,45,'35-45'), (45,55,'45-55'), (55,70,'55-70'),
                        (70,90,'70-90'), (90,120,'90-120'), (120,10**6,'120+')]:
        m = (px >= lo) & (px < hi)
        if not m.sum(): continue
        r = ok[m].mean()
        print(f"    {lab:>8} px  n={int(m.sum()):>4}  recall {r:>6.1%}")
        if r >= 0.80 and pxl is None: pxl = lo
    return env, prec

env284, _ = report(THRESHOLDS[0])
print()
env40, _ = report(THRESHOLDS[1])

# ---------------- the range spec ----------------
def px_for_recall(tgt):
    """smallest face-pixel size whose bucket still meets the recall target (at THRESHOLDS[0])"""
    acc = best >= THRESHOLDS[0]; ok = acc & correct
    for lo, hi in [(0,35),(35,45),(45,55),(55,70),(70,90),(90,120),(120,10**6)]:
        m = (px >= lo) & (px < hi)
        if m.sum() and ok[m].mean() >= tgt:
            return lo
    return None

print("\n" + "=" * 84)
print("RANGE SPEC — what your optics must deliver")
print("=" * 84)
for tgt in RECALL_TARGETS:
    need = px_for_recall(tgt)
    if need is None:
        print(f"  recall {tgt:.0%}: not achieved at any face size in this data"); continue
    print(f"\n  To hold {tgt:.0%} recall you need a face of at least ~{need} px.")
    print(f"    On DroneFace's camera (k={K_DRONEFACE:.0f}) that is {K_DRONEFACE/need:.1f} m.")
    for R in TARGET_RANGES:
        print(f"    To reach {R:.0f} m your camera needs k >= {need*R:.0f} "
              f"({need*R/K_DRONEFACE:.1f}x DroneFace's optics)")
    if K_YOUR_DRONE:
        print(f"    YOUR drone (k={K_YOUR_DRONE:.0f}) reaches {K_YOUR_DRONE/need:.1f} m")

print("\n  k is a single constant of your camera+lens: face_px = k / distance_m.")
print("  Calibrate it once — stand someone at a known distance D, read the face px from")
print("  21_run_folder.py's overlay, then k = D * face_px. Set K_YOUR_DRONE above and re-run")
print("  to convert every pixel limit into metres for YOUR aircraft.")

# ---------------- plot ----------------
pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("envelope_%H%M%S"))
os.makedirs(pdir, exist_ok=True)
dd = sorted(set(dist))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for thr, col in [(THRESHOLDS[0], '#0C7C88'), (THRESHOLDS[1], '#7f8c8d')]:
    okt = (best >= thr) & correct
    ax[0].plot(dd, [okt[dist == d].mean() for d in dd], 'o-', color=col, lw=2, label=f'thr {thr}')
ax[0].axhline(0.80, ls=':', color='#C0392B'); ax[0].axhline(0.50, ls=':', color='#B67611')
ax[0].set_xlabel('distance (m)'); ax[0].set_ylabel('recall'); ax[0].set_ylim(0, 1.02)
ax[0].set_title('Recall vs distance (DroneFace optics)'); ax[0].grid(alpha=.3); ax[0].legend()
okt = (best >= THRESHOLDS[0]) & correct
for h, c in zip(sorted(set(hgt)), plt.cm.viridis(np.linspace(0, .85, len(set(hgt))))):
    m = hgt == h
    ax[1].plot(dd, [okt[m & (dist == d)].mean() for d in dd], '-', color=c, lw=1.8, label=f'{h:g} m')
ax[1].axhline(0.80, ls=':', color='#C0392B')
ax[1].set_xlabel('distance (m)'); ax[1].set_ylabel('recall'); ax[1].set_ylim(0, 1.02)
ax[1].set_title(f'By camera height (thr {THRESHOLDS[0]})'); ax[1].grid(alpha=.3); ax[1].legend(title='height')
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'distance_envelope.png'), dpi=150)
print(f"\n[SAVED] plot -> {pdir}")
