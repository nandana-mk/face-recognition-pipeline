"""
29_calibrate_scface.py — validate FaceRecognizer.calibrate_threshold() against a KNOWN answer.

    python 29_calibrate_scface.py "C:\\dev\\face-recognition-pipeline\\SCface_database"

PURPOSE - this is a TEST OF THE METHOD, not a way to get your deployment threshold.
    27_eval_scface.py already computed the correct SCface threshold by an independent route:
        z_threshold = 9.727  at 0% stranger false-accepts, recall 67.5%
    If calibrate_threshold() reproduces that number from the same data, the function is correct
    and you can trust it on YOUR data. If it does not, something is wrong and we fix it here
    rather than discovering it in the field.

    Your real deployment threshold must come from YOUR watchlist and YOUR strangers.
    SCface has 98 identities on indoor CCTV; you have ~9 on a drone. Thresholds do not transfer -
    that is precisely what this whole exercise established.

It also rehearses the exact folder layout you will use for your own calibration:
    <work>/gallery/<ID>/     one reference image per identity   -> known_faces_dir
    <work>/genuine/<ID>/     probe images of enrolled people    -> genuine_dir
    <work>/strangers/        images of NON-enrolled people      -> stranger_dir
"""
import os, re, sys, glob, shutil
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_recognition_module import FaceRecognizer, Config      # noqa: E402

WORK       = r"C:\dev\train_val\calib_scface"
COHORT_NPZ = r"C:\dev\train_val\models\cohort.npz"
TORCH_LIB  = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
N_ENROLL   = 100          # same split as 27_eval_scface.py
SEED       = 7            # same seed, so the comparison is apples-to-apples
USE_IR     = False
EXPECTED   = 9.727        # what 27_eval_scface.py found at 0% FAR
IMG_EXT    = ('.jpg', '.jpeg', '.png', '.bmp')

root = sys.argv[1] if len(sys.argv) > 1 else None
if not root or not os.path.isdir(root):
    print("usage: python 29_calibrate_scface.py <SCface root>"); raise SystemExit

# ---------------- discover (same parsing as 27) ----------------
files = [p for p in glob.glob(os.path.join(root, '**', '*'), recursive=True)
         if p.lower().endswith(IMG_EXT)]
files.sort(key=lambda p: (0 if 'cropped' in p.lower() else 1,
                          0 if 'surveillance_cameras_all' in p.replace('\\', '/') else 1, p))
re_surv = re.compile(r'(\d{3})_cam(\d+)_(\d)', re.I)
re_mug = re.compile(r'(\d{3})[_\-]?(frontal|mugshot)', re.I)
mug, surv, seen = {}, [], set()
for p in files:
    b, low = os.path.basename(p), p.lower()
    if 'rotation' in low:
        continue
    m = re_mug.search(b)
    if m and ('mug' in low or 'frontal' in b.lower()):
        mug.setdefault(m.group(1), p); continue
    s = re_surv.search(b)
    if s:
        if not USE_IR and int(s.group(2)) > 5: continue
        if b.lower() in seen: continue
        seen.add(b.lower()); surv.append((s.group(1), p))
print(f"[DISCOVER] {len(mug)} mugshots, {len(surv)} surveillance probes")

sids = sorted(mug)
rng = np.random.default_rng(SEED); rng.shuffle(sids)
enrolled, strangers = set(sids[:N_ENROLL]), set(sids[N_ENROLL:])
print(f"[SPLIT] {len(enrolled)} enrolled, {len(strangers)} strangers (seed {SEED})")

# ---------------- build the calibration folder layout ----------------
if os.path.isdir(WORK):
    shutil.rmtree(WORK)
for sub in ('gallery', 'genuine', 'strangers'):
    os.makedirs(os.path.join(WORK, sub), exist_ok=True)
for sid in enrolled:
    d = os.path.join(WORK, 'gallery', sid); os.makedirs(d, exist_ok=True)
    shutil.copy2(mug[sid], os.path.join(d, os.path.basename(mug[sid])))
n_gen = n_str = 0
for sid, p in surv:
    if sid in enrolled:
        d = os.path.join(WORK, 'genuine', sid); os.makedirs(d, exist_ok=True)
        shutil.copy2(p, os.path.join(d, os.path.basename(p))); n_gen += 1
    elif sid in strangers:
        shutil.copy2(p, os.path.join(WORK, 'strangers', os.path.basename(p))); n_str += 1
print(f"[PREP] gallery {len(enrolled)} refs | genuine {n_gen} probes | strangers {n_str} probes")
print(f"[PREP] layout -> {WORK}\n")

# ---------------- run the module exactly as the GCS would ----------------
fr = FaceRecognizer(Config(
    known_faces_dir=os.path.join(WORK, 'gallery'),
    cohort_npz=COHORT_NPZ,
    torch_lib_dir=TORCH_LIB,
    min_frames=1,                 # stills, not tracks
))
print()
res = fr.calibrate_threshold(
    genuine_dir=os.path.join(WORK, 'genuine'),
    stranger_dir=os.path.join(WORK, 'strangers'),
    target_far=0.0,               # 0% stranger false-accepts, matching 27_eval_scface.py
    apply=False,
)

print("\n" + "=" * 72)
print("VALIDATION against the independent result from 27_eval_scface.py")
print("=" * 72)
print(f"  27_eval_scface.py  z_threshold = {EXPECTED:.3f}   recall 67.5%   FAR 0.00%")
print(f"  calibrate_threshold z_threshold = {res.threshold:.3f}   "
      f"recall {res.recall:.1%}   FAR {res.actual_far:.2%}")
delta = abs(res.threshold - EXPECTED)
if delta < 0.5 and abs(res.recall - 0.675) < 0.05:
    print(f"\n  >>> MATCH (delta {delta:.3f}). calibrate_threshold() is correct.")
    print("      Now run it on YOUR OWN watchlist + strangers to get the deployment threshold.")
else:
    print(f"\n  >>> MISMATCH (delta {delta:.3f}). Do not trust the method yet - tell me the")
    print("      numbers and I will find the discrepancy before you calibrate on real data.")
print("\n  REMINDER: this threshold is for SCface's 98-identity CCTV gallery. It is NOT yours.")
