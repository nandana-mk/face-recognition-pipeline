"""
28_compare_labeled.py — OLD vs NEW on YOUR OWN labelled data, with YOUR OWN strangers.

    python 26_build_cohort.py "<lfw folder>"     # first, if cohort.npz does not exist yet
    python 28_compare_labeled.py

WHY THIS IS THE BEST TEST AVAILABLE TODAY
  labeled_eval.csv indexes 3,288 crops from your real drone footage:
     2,382 GENUINE  crops of enrolled watchlist people
       906 IMPOSTOR crops of people NOT on the watchlist (Sakthi, Jaga, Harsh, ...)
  Same cameras, same lighting, same distances, same population as deployment. The false-accept
  rate measured here is far more trustworthy than the LFW-celebrity number (1 in 1,500), because
  these strangers are the actual people who walk through your frames.

  Enrollment uses ONE reference photo per person from Known_faces - your real constraint.

WHAT IS COMPARED
  A  OLD : raw cosine >= THRESHOLD
  B  NEW : AS-Norm z >= z_thresh calibrated on DroneFace   (does the calibration transfer?)
  C  NEW : AS-Norm z re-tuned so stranger false-accepts EXACTLY match A   (the fair comparison)

Only A vs C is a fair head-to-head: equal precision cost, measured recall difference.

Outputs: console tables (overall + per condition + per identity) and reports/compare_labeled.csv
"""
import os, csv
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np
from collections import defaultdict
from insightface.app import FaceAnalysis
from insightface.utils import face_align

LABELED    = r"C:\dev\train_val\reports\labeled_eval.csv"
KNOWN_DIR  = r"C:\dev\face-recognition-pipeline\Known_faces"
COHORT_NPZ = r"C:\dev\train_val\models\cohort.npz"
OUT_CSV    = r"C:\dev\train_val\reports\compare_labeled.csv"
THRESHOLD  = 0.40
USE_DENOISE = True
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def embed_batch(c):
    f = rec.get_feat(c); f = np.asarray(f, np.float32).reshape(len(c), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)

def prep(img):
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    return cv2.bilateralFilter(img, 5, 50, 50) if USE_DENOISE else img

# ---------------- gallery: ONE reference per person ----------------
names, gvec = [], []
for entry in sorted(os.listdir(KNOWN_DIR)):
    full = os.path.join(KNOWN_DIR, entry)
    if os.path.isdir(full) and not entry.startswith('_'):
        person = entry
        fs = [os.path.join(full, f) for f in sorted(os.listdir(full)) if f.lower().endswith(IMG_EXT)]
    elif entry.lower().endswith(IMG_EXT):
        person = os.path.splitext(entry)[0]; fs = [full]
    else:
        continue
    vs = []
    for p in fs:
        im = cv2.imread(p)
        if im is None: continue
        bb, kps = det.detect(im, max_num=0, metric='default')
        if bb is None or len(bb) == 0: continue
        k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        vs.append(embed_batch([prep(face_align.norm_crop(im, kps[k]))])[0])
    if not vs: print(f"  [skip] {person}"); continue
    m = np.mean(vs, 0); m /= (np.linalg.norm(m) + 1e-9)
    names.append(person); gvec.append(m.astype(np.float32))
G = np.array(gvec, np.float32); names = np.array(names)
print(f"[ENROLL] {len(names)}: {', '.join(names)}\n")

# ---------------- load the labelled crops ----------------
rows = []
with open(LABELED, newline='', encoding='utf-8') as fh:
    for r in csv.DictReader(fh): rows.append(r)
def truthy(v): return str(v).strip().lower() in ('true', '1', 'yes')

crops, meta, missing = [], [], 0
for r in rows:
    p = r['path']
    im = cv2.imread(p)
    if im is None: missing += 1; continue
    crops.append(prep(im))
    meta.append((r['cond'], r['true'], truthy(r['impostor'])))
if missing: print(f"[WARN] {missing} crops listed in the CSV are missing on disk")
P = np.concatenate([embed_batch(crops[i:i+256]) for i in range(0, len(crops), 256)], 0)
cond = np.array([m[0] for m in meta]); true = np.array([m[1] for m in meta])
is_imp = np.array([m[2] for m in meta], bool)
print(f"[PROBES] {len(crops)} crops | {(~is_imp).sum()} genuine, {is_imp.sum()} impostor\n")

# ---------------- scores ----------------
S_raw = P @ G.T
S_z = None
if os.path.exists(COHORT_NPZ):
    _c = np.load(COHORT_NPZ, allow_pickle=True)
    COH = _c['cohort'].astype(np.float32); TOPK = int(_c['topk']); ZT = float(_c['z_thresh'])
    c = P @ COH.T
    t = np.partition(c, -TOPK, axis=1)[:, -TOPK:]
    S_z = (S_raw - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)
    print(f"[AS-Norm] cohort {COH.shape[0]}, K={TOPK}, DroneFace z_thresh={ZT:.3f}\n")
else:
    print(f"[ERROR] no cohort at {COHORT_NPZ} - run 26_build_cohort.py first"); raise SystemExit

def evaluate(S, thr):
    top = names[S.argmax(1)]; best = S.max(1); acc = best >= thr
    gen = ~is_imp
    tp = int((acc & gen & (top == true)).sum())
    fp_name = int((acc & gen & (top != true)).sum())
    fp_imp = int((acc & is_imp).sum())
    prec = tp / max(tp + fp_name + fp_imp, 1)
    rec = tp / max(gen.sum(), 1)
    return dict(thr=thr, tp=tp, fp_name=fp_name, fp_imp=fp_imp, prec=prec, rec=rec,
                f1=2*prec*rec/max(prec+rec, 1e-9), far=fp_imp/max(is_imp.sum(), 1),
                ok=acc & gen & (top == true))

A = evaluate(S_raw, THRESHOLD)
B = evaluate(S_z, ZT)
# C: match A's stranger false-accept count exactly
imp_sorted = np.sort(S_z[is_imp].max(1))
k = A['fp_imp']
thr_c = float(imp_sorted[len(imp_sorted)-k-1]) + 1e-6 if k < len(imp_sorted) else float(imp_sorted[-1]) + 1e-6
C = evaluate(S_z, thr_c)

print("=" * 88)
print("HEAD-TO-HEAD on your own footage, with your own strangers")
print("=" * 88)
print(f"  {'configuration':<34} {'thr':>8} {'precision':>10} {'recall':>8} {'F1':>7} "
      f"{'stranger-FA':>12} {'wrong-name':>11}")
for nm, r in [("A  OLD raw cosine", A), ("B  NEW AS-Norm (DroneFace thr)", B),
              ("C  NEW AS-Norm (matched FA)", C)]:
    print(f"  {nm:<34} {r['thr']:>8.3f} {r['prec']:>10.4f} {r['rec']:>8.1%} {r['f1']:>7.4f} "
          f"{r['far']:>11.2%} {r['fp_name']:>11d}")

print(f"\n  FAIR COMPARISON (A vs C — identical stranger false-accepts: {A['fp_imp']}):")
print(f"     recall     {A['rec']:.1%}  ->  {C['rec']:.1%}   ({(C['rec']-A['rec'])*100:+.1f} pts)")
print(f"     precision  {A['prec']:.4f}  ->  {C['prec']:.4f}")
print(f"     wrong-name accepts {A['fp_name']}  ->  {C['fp_name']}   "
      f"<-- watch this: AS-Norm can also promote WRONG names")
verdict = "WINS" if (C['rec'] > A['rec'] + 0.01 and C['fp_name'] <= A['fp_name'] + 5) else \
          "is NOT clearly better"
print(f"     VERDICT: AS-Norm {verdict} on your own data")

print(f"\n  by condition:")
print(f"    {'condition':<44} {'n':>5} {'OLD':>8} {'NEW':>8} {'delta':>8}")
for cd in sorted(set(cond)):
    m = (cond == cd) & ~is_imp
    if not m.sum(): continue
    a, b = A['ok'][m].mean(), C['ok'][m].mean()
    print(f"    {cd[:44]:<44} {int(m.sum()):>5} {a:>7.1%} {b:>7.1%} {b-a:>+8.1%}")

print(f"\n  by identity:")
print(f"    {'person':<16} {'n':>5} {'OLD':>8} {'NEW':>8} {'delta':>8}")
for pn in sorted(set(true[~is_imp])):
    m = (true == pn) & ~is_imp
    if not m.sum(): continue
    a, b = A['ok'][m].mean(), C['ok'][m].mean()
    print(f"    {pn:<16} {int(m.sum()):>5} {a:>7.1%} {b:>7.1%} {b-a:>+8.1%}")

with open(OUT_CSV, 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['cond', 'true', 'is_impostor', 'pred', 'sim', 'z', 'ok_old', 'ok_new'])
    topr = names[S_raw.argmax(1)]
    for i in range(len(meta)):
        w.writerow([cond[i], true[i], int(is_imp[i]), topr[i], round(float(S_raw[i].max()), 3),
                    round(float(S_z[i].max()), 2), int(A['ok'][i]), int(C['ok'][i])])
print(f"\n[SAVED] {OUT_CSV}")
