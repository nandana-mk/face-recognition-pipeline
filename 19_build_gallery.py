"""
19_build_gallery.py — build a QUALITY-GATED multi-reference gallery and save it for fr2/10 to load.

Reconciles the two findings:
  DroneFace: multi-reference enrollment lifted recall 16% -> 54% (clean, varied refs)
  Earlier  : averaging NOISY video crops DEGRADED the batch (muddied template)
=> add several references per person BUT reject blurry/small/low-confidence ones before averaging.

Input  (either layout works, per person):
  Known_faces/<Name>.jpg                      (single photo — still works, just no multi gain)
  Known_faces/<Name>/*.jpg                    (RECOMMENDED — several clean, varied shots)
Output:
  models/gallery.npz  ->  names[], embs[]  (one L2-normalised MEAN template per person; drop-in for `G @ e`)
  reports/gallery_audit.txt / contact info in console

Quality gates (a reference is used only if it passes all):
  face detected · face px >= MIN_PX · blur(Laplacian var) >= MIN_BLUR · feat-norm >= MIN_NORM
Also prints a confusability warning if two people's templates are closer than CONFUSE_MAX.
"""
import os
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

KNOWN_DIR = r"C:\dev\face-recognition-pipeline\Known_faces"
OUT_NPZ   = r"C:\dev\train_val\models\gallery.npz"
EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
MIN_PX, MIN_BLUR, MIN_NORM, CONFUSE_MAX = 60, 80.0, 12.0, 0.45
USE_DENOISE = True      # match fr2's get_embedding

os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def embed_aligned(crop112):
    img = cv2.bilateralFilter(crop112, 5, 50, 50) if USE_DENOISE else crop112
    f = rec.get_feat(img).flatten().astype(np.float32)
    return f, float(np.linalg.norm(f) + 1e-9)

def process(path):
    """return (unit_emb, px, blur, norm) or None with a reason."""
    im = cv2.imread(path)
    if im is None: return None, "unreadable"
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0: return None, "no-face"
    k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
    px = int(min(bb[k, 2]-bb[k, 0], bb[k, 3]-bb[k, 1]))
    crop = face_align.norm_crop(im, kps[k])
    blur = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    f, nrm = embed_aligned(crop)
    reasons = []
    if px < MIN_PX: reasons.append(f"small({px}px)")
    if blur < MIN_BLUR: reasons.append(f"blurry({blur:.0f})")
    if nrm < MIN_NORM: reasons.append(f"lownorm({nrm:.0f})")
    return (f / nrm, px, blur, nrm), (";".join(reasons) if reasons else "ok")

# ---------- gather per-person reference files ----------
people = {}     # name -> [file paths]
for entry in sorted(os.listdir(KNOWN_DIR)):
    full = os.path.join(KNOWN_DIR, entry)
    if os.path.isdir(full):
        if entry.startswith("_"): continue
        fs = [os.path.join(full, f) for f in sorted(os.listdir(full)) if f.lower().endswith(EXT)]
        if fs: people[entry] = fs
    elif entry.lower().endswith(EXT):
        people[os.path.splitext(entry)[0]] = [full]

print(f"[INFO] {len(people)} identities found\n")
names, embs = [], []
print(f"  {'person':14s} {'used':>4} {'skip':>4}  detail")
for name, files in people.items():
    used, skipped = [], []
    for p in files:
        res, why = process(p)
        if res is None or why != "ok":
            skipped.append((os.path.basename(p), why));
            if res is None: continue
            # keep quality info but do NOT use a flagged crop in the average
            continue
        used.append(res[0])
    if not used:
        print(f"  {name:14s} {0:>4} {len(skipped):>4}  [!] NO usable reference — recapture this person")
        continue
    mean = np.mean(used, axis=0); mean /= (np.linalg.norm(mean) + 1e-9)
    names.append(name); embs.append(mean.astype(np.float32))
    detail = "" if not skipped else "skipped: " + ", ".join(f"{f}({w})" for f, w in skipped[:3])
    print(f"  {name:14s} {len(used):>4} {len(skipped):>4}  {detail}")

if not names:
    print("\n[ERROR] no gallery built."); raise SystemExit
E = np.array(embs, np.float32)

# ---------- confusability audit ----------
S = E @ E.T; np.fill_diagonal(S, -1)
print("\n  Confusability (nearest other person):")
warn = False
for i, n in enumerate(names):
    j = int(np.argmax(S[i])); s = float(S[i, j])
    flag = "  <-- TOO CLOSE" if s > CONFUSE_MAX else ""
    if s > CONFUSE_MAX: warn = True
    print(f"    {n:14s} nearest={names[j]:14s} sim={s:.3f}{flag}")
if warn:
    print("  [!] Flagged pairs risk cross-identity false accepts — improve those references or enable a margin gate.")

np.savez(OUT_NPZ, names=np.array(names), embs=E)
print(f"\n[SAVED] gallery -> {OUT_NPZ}   ({len(names)} identities, {E.shape[1]}-d)")
print("Next: point fr2.py / 10_batch_report.py at this file instead of enrolling a single photo (see integration snippet).")
