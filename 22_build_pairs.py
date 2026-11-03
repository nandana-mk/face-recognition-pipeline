"""
22_build_pairs.py — build REALISTIC (reference, drone-probe) embedding pairs.

    python 22_build_pairs.py "C:\\Users\\hticd\\scikit_learn_data\\lfw_home\\lfw_funneled"

WHAT CHANGED vs the first version (which produced a useless baseline of 0.776):

  1. CROSS-PHOTO PAIRS, not self-pairs.
     Before: target = the SAME photo, clean. cos started at 1.0 and only blur moved it -> too easy.
     Now:    target = a DIFFERENT photo of the same person (the "enrollment mugshot"), and the
             probe is a degraded OTHER photo. This is the actual deployment task: one reference
             photo vs a drone probe. On DroneFace, portrait-vs-aerial is only ~0.43 even CLEAN,
             so cross-photo variation is most of the difficulty.

  2. DEGRADE BEFORE DETECTION, not after alignment.
     Before: align on the full-res image (perfect landmarks) then blur the 112 crop -> no
             alignment error at all.
     Now:    shrink the WHOLE image so the face is only k px, apply sensor noise/blur/jpeg, then
             RE-DETECT and align on that degraded image. The landmarks are therefore estimated
             from a small noisy face exactly as they are in flight. Misalignment is a large part
             of real far-face damage and was entirely missing before.

Only people with >= 2 photos are usable (LFW: ~1,680 identities / ~9,164 images). That is thin -
fine to prototype, but CASIA-WebFace is the proper training set if the gates come back marginal.

Output: models/pairs.npz
    deg[M,512]  probe embeddings (degraded, re-detected, misaligned)
    tgt[M,512]  the reference embedding each probe must be pulled toward
    pid[M]      person id per pair (used to split train/val BY PERSON - no leakage)
    clean[P,512], clean_pid[P]   clean reference embeddings (for the preserve term)
"""
import os, sys
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

OUT_NPZ   = r"C:\dev\train_val\models\pairs.npz"
K_PER_IMG = 4              # degraded probes per source photo (each needs its own re-detection)
MIN_IMGS  = 2              # a person needs >=2 photos: one as reference, others as probes
MAX_PERSON_IMGS = 12       # cap prolific identities so they do not dominate
IMG_EXT   = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.ppm')
SEED      = 0

rng = np.random.default_rng(SEED)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def embed_batch(crops):
    f = rec.get_feat(crops)
    f = np.asarray(f, np.float32).reshape(len(crops), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)

def detect_largest(im):
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0: return None, None, 0.0
    j = int(np.argmax((bb[:, 2]-bb[:, 0]) * (bb[:, 3]-bb[:, 1])))
    w = float(min(bb[j, 2]-bb[j, 0], bb[j, 3]-bb[j, 1]))
    return bb[j], kps[j], w

def pitch_warp(img, amt):
    h = w = 112
    dx, dy = amt * w * 0.5, amt * h * 0.25
    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    dst = np.float32([[0, dy], [w-1, dy], [w-1-dx, h-1], [dx, h-1]])
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderMode=cv2.BORDER_REPLICATE)

def make_probe(img, kps_clean, face_w, r):
    """Shrink the WHOLE image so the face is k px, corrupt it, then RE-DETECT and align.
       This reproduces the real failure chain: small face -> bad landmarks -> misaligned crop."""
    k = int(r.integers(28, 81))
    s = k / max(face_w, 1e-6)
    H, W = img.shape[:2]
    nh, nw = max(16, int(H*s)), max(16, int(W*s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    # sensor-level corruption, applied to the small image (before detection)
    if r.random() < 0.6:
        kk = int(r.choice([3, 5]))
        small = cv2.GaussianBlur(small, (kk, kk), 0)
    if r.random() < 0.75:
        small = np.clip(small.astype(np.float32) + r.normal(0, r.uniform(2, 14), small.shape),
                        0, 255).astype(np.uint8)
    if r.random() < 0.6:
        small = np.clip(small.astype(np.float32) * r.uniform(0.35, 1.0), 0, 255).astype(np.uint8)
    if r.random() < 0.5:
        ok, enc = cv2.imencode('.jpg', small, [int(cv2.IMWRITE_JPEG_QUALITY), int(r.integers(30, 96))])
        if ok: small = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # re-detect on the degraded small image -> realistic landmark error
    _, kps_s, _ = detect_largest(small)
    if kps_s is None:
        kps_s = kps_clean * s + r.normal(0, max(0.6, 0.035*k), kps_clean.shape)   # jitter fallback
    crop = face_align.norm_crop(small, kps_s.astype(np.float32))
    if r.random() < 0.5:
        crop = pitch_warp(crop, float(r.uniform(0.05, 0.28)))
    return crop

# ---------------- gather people ----------------
src = sys.argv[1] if len(sys.argv) > 1 else None
if not src or not os.path.isdir(src):
    print("usage: python 22_build_pairs.py <dataset root with one folder per person>")
    raise SystemExit

people = {}
for d in sorted(os.listdir(src)):
    pdir = os.path.join(src, d)
    if not os.path.isdir(pdir): continue
    fs = [os.path.join(pdir, f) for f in sorted(os.listdir(pdir)) if f.lower().endswith(IMG_EXT)]
    if len(fs) >= MIN_IMGS: people[d] = fs[:MAX_PERSON_IMGS]
print(f"[INFO] {len(people)} identities with >= {MIN_IMGS} photos "
      f"({sum(len(v) for v in people.values())} images)")
if len(people) < 500:
    print("[WARN] few identities. This will prototype but may not generalise; "
          "CASIA-WebFace (10.5k IDs) is the proper set.")

# ---------------- build pairs ----------------
deg_v, tgt_v, pid_v, clean_v, cpid_v = [], [], [], [], []
buf_crop, buf_tgt, buf_pid = [], [], []

def flush():
    global buf_crop, buf_tgt, buf_pid
    if not buf_crop: return
    deg_v.append(embed_batch(buf_crop)); tgt_v.append(np.array(buf_tgt, np.float32))
    pid_v.extend(buf_pid); buf_crop, buf_tgt, buf_pid = [], [], []

n_person = 0
for pi, (person, files) in enumerate(people.items()):
    # embed every photo cleanly; pick the largest-face one as the enrollment reference
    crops, kpss, ws, imgs = [], [], [], []
    for p in files:
        im = cv2.imread(p)
        if im is None: continue
        bb, kps, w = detect_largest(im)
        if kps is None or w < 50: continue
        crops.append(face_align.norm_crop(im, kps)); kpss.append(kps); ws.append(w); imgs.append(im)
    if len(crops) < 2: continue
    E = embed_batch(crops)
    ref = int(np.argmax(ws))                       # biggest, cleanest face = the "mugshot"
    ref_emb = E[ref]
    clean_v.append(ref_emb); cpid_v.append(n_person)
    for i in range(len(crops)):
        if i == ref: continue                      # probe must be a DIFFERENT photo
        for _ in range(K_PER_IMG):
            buf_crop.append(make_probe(imgs[i], kpss[i], ws[i], rng))
            buf_tgt.append(ref_emb); buf_pid.append(n_person)
    n_person += 1
    if len(buf_crop) >= 256: flush()
    if (pi + 1) % 200 == 0:
        print(f"  ...{pi+1}/{len(people)} identities, {sum(len(x) for x in deg_v)} pairs")
flush()

DEG = np.concatenate(deg_v, 0).astype(np.float32)
TGT = np.concatenate(tgt_v, 0).astype(np.float32)
PID = np.array(pid_v, np.int32)
CLEAN = np.array(clean_v, np.float32); CPID = np.array(cpid_v, np.int32)
print(f"\n[INFO] {n_person} identities -> {DEG.shape[0]} (probe, reference) pairs")

base = (DEG * TGT).sum(1)
print(f"[INFO] baseline cos(probe, reference) BEFORE adapter: "
      f"mean {base.mean():.3f}  median {np.median(base):.3f}  frac<0.40 {np.mean(base < 0.40):.1%}")
print("       For reference, real DroneFace genuine sims at 45-70px are 0.26-0.31.")
if base.mean() > 0.6:
    print("       [!] still too easy - degradation is not biting. Do not train on this.")
elif np.mean(base < 0.40) < 0.15:
    print("       [!] very few hard cases - consider harsher settings.")
else:
    print("       -> realistic difficulty. Good to train.")

os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez_compressed(OUT_NPZ, deg=DEG, tgt=TGT, pid=PID, clean=CLEAN, clean_pid=CPID)
print(f"[SAVED] {OUT_NPZ}")
print("Next: python 23_train_adapter.py")
