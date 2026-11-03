"""
18_multiref_gallery.py — PROVE whether multi-reference enrollment lifts the numbers (leakage-free).

The forensics showed even a clean DroneFace face matches its single portrait at only ~0.43. Hypothesis:
a richer gallery (several shots per person) raises genuine similarity everywhere. This tests it FAIRLY:

  ENROLL only from camera heights gp 0 (1.5m) and gp 3 (3.0m).
  TEST  only on camera heights gp 4 (4.0m) and gp 5 (5.0m)  -> completely disjoint, no crop is in both.

Three galleries compared on the SAME held-out test set:
  single      : portrait only (today's method)
  multi_mean  : mean of {portrait + up to K clean near crops}          -> averaged template
  multi_max   : keep all {portrait + K crops} as separate templates, score = max over them
For each: closed-set recall@0.40, mean genuine sim, mean margin (genuine - best impostor), and the
impostor floor (so you can see whether multi_max's recall gain costs precision).
"""
import os
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, csv, datetime
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
THRESHOLD = 0.40
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
ENROLL_GP = {'0', '3'}       # heights 1.5m, 3.0m  -> enrollment source
TEST_GP   = {'4', '5'}       # heights 4.0m, 5.0m  -> held-out test
ENROLL_MIN_PX = 80           # only enroll from reasonably large crops
K_EXTRA = 4                  # up to this many near crops added to the portrait

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def unit(img):
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    f = rec.get_feat(img).flatten().astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-9)

def align_from_photo(path):
    im = cv2.imread(path)
    if im is None: return None
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is not None and len(bb) > 0:
        k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        return face_align.norm_crop(im, kps[k])
    return cv2.resize(im, (112, 112))

def find_frontal(subj):
    for folder, prefix in [(PORT, f"{subj}_"), (TRIO_C, f"{subj}_cam_0_porf"), (TRIO_G, f"{subj}_gp_0_porf")]:
        if not os.path.isdir(folder): continue
        for f in sorted(os.listdir(folder)):
            if f.lower().startswith(prefix) and f.lower().endswith(IMG_EXT):
                return os.path.join(folder, f)
    return None

subjects = sorted({f.split('_')[0] for f in os.listdir(FACES) if f.lower().endswith(IMG_EXT)})

# ---------- build enrollment pool (portrait + near crops from ENROLL_GP) and test set (TEST_GP) ----------
enroll_pool = defaultdict(list)     # sid -> [unit emb, ...]  (portrait first)
test_rows = []                      # (sid, px, img)
for s in subjects:
    fp = find_frontal(s)
    if fp:
        c = align_from_photo(fp)
        if c is not None: enroll_pool[s].append(unit(c))

cand = defaultdict(list)            # sid -> [(px, emb)] from ENROLL_GP
for f in os.listdir(FACES):
    if not f.lower().endswith(IMG_EXT): continue
    p = f.split('_')
    if len(p) < 5 or p[1] != 'gp': continue
    sid, gp, seq = p[0], p[2], p[4].split('.')[0]
    im = cv2.imread(os.path.join(FACES, f))
    if im is None: continue
    px = min(im.shape[0], im.shape[1])
    if gp in ENROLL_GP and px >= ENROLL_MIN_PX:
        cand[sid].append((px, unit(im)))
    elif gp in TEST_GP:
        test_rows.append((sid, px, 17 - int(seq)/2.0, im))
for s in subjects:
    for _, e in sorted(cand[s], key=lambda t: -t[0])[:K_EXTRA]:
        enroll_pool[s].append(e)

names = [s for s in subjects if enroll_pool[s]]
print(f"[INFO] enrolled {len(names)} subjects; pool sizes: " +
      ", ".join(f"{s}:{len(enroll_pool[s])}" for s in names))
print(f"[INFO] held-out test crops (heights 4-5m): {len(test_rows)}")

# ---------- three galleries ----------
G_single = {s: enroll_pool[s][0] for s in names}                                  # portrait only
G_mean = {}
for s in names:
    m = np.mean(enroll_pool[s], axis=0); G_mean[s] = m / (np.linalg.norm(m) + 1e-9)
G_max = {s: np.array(enroll_pool[s], np.float32) for s in names}                  # all templates

def score_single(e, gal): return {s: float(e @ gal[s]) for s in names}
def score_mean(e, gal):   return {s: float(e @ gal[s]) for s in names}
def score_max(e, gal):    return {s: float(np.max(gal[s] @ e)) for s in names}

SCHEMES = [('single', G_single, score_single), ('multi_mean', G_mean, score_mean), ('multi_max', G_max, score_max)]

# ---------- evaluate ----------
def eval_scheme(gal, scorer):
    n = len(test_rows); tp = 0; gsum = 0.0; msum = 0.0; impsum = 0.0
    by_d = defaultdict(lambda: [0, 0])   # dist -> [correct, n]
    for sid, px, dist, im in test_rows:
        e = unit(im)
        sc = scorer(e, gal)
        gen = sc[sid]
        best = max(sc, key=sc.get); bestsim = sc[best]
        imp = max(v for k, v in sc.items() if k != sid)
        gsum += gen; impsum += imp; msum += (gen - imp)
        ok = (best == sid and bestsim >= THRESHOLD)
        tp += int(ok)
        by_d[dist][1] += 1; by_d[dist][0] += int(ok)
    return dict(recall=tp/n, mean_genuine=gsum/n, mean_impostor=impsum/n, mean_margin=msum/n, by_d=by_d)

print("\n" + "=" * 66)
print(f"HELD-OUT TEST (enroll @1.5/3.0m  ->  test @4.0/5.0m), thr={THRESHOLD}")
print("=" * 66)
print(f"  {'scheme':>11} {'recall':>8} {'mean-gen':>9} {'mean-imp':>9} {'margin':>8}")
res = {}
for label, gal, scorer in SCHEMES:
    r = eval_scheme(gal, scorer); res[label] = r
    print(f"  {label:>11} {r['recall']:>7.1%} {r['mean_genuine']:>9.3f} {r['mean_impostor']:>9.3f} {r['mean_margin']:>+8.3f}")

base = res['single']['recall']
print(f"\n  multi_mean recall lift vs single: {(res['multi_mean']['recall']-base)*100:+.1f} pts")
print(f"  multi_max  recall lift vs single: {(res['multi_max']['recall']-base)*100:+.1f} pts")
print(f"  NOTE: watch mean-imp — if multi_max raises impostor sim a lot, its recall gain costs precision.")

# ---------- plot recall vs distance per scheme ----------
pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("multiref_%H%M%S"))
os.makedirs(pdir, exist_ok=True)
COL = {'single': '#7f8c8d', 'multi_mean': '#0C7C88', 'multi_max': '#B67611'}
fig, ax = plt.subplots(figsize=(9.5, 5.5))
for label in ['single', 'multi_mean', 'multi_max']:
    bd = res[label]['by_d']; xs = sorted(bd)
    ax.plot(xs, [bd[x][0]/bd[x][1] for x in xs], 'o-', color=COL[label], lw=2, label=label)
ax.axhline(0, color='k', lw=.5)
ax.set_xlabel('distance (m)'); ax.set_ylabel('recall (held-out heights 4-5m)'); ax.set_ylim(0, 1.02)
ax.set_title('Does multi-reference enrollment lift recall? (leakage-free)')
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'multiref_recall.png'), dpi=150); plt.close(fig)

with open(os.path.join(OUT_DIR, "multiref_compare.csv"), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['scheme', 'recall', 'mean_genuine', 'mean_impostor', 'mean_margin'])
    for label in ['single', 'multi_mean', 'multi_max']:
        r = res[label]; w.writerow([label, r['recall'], r['mean_genuine'], r['mean_impostor'], r['mean_margin']])
print(f"\n[SAVED] plot -> {pdir}\\multiref_recall.png")
print(f"[SAVED] {os.path.join(OUT_DIR, 'multiref_compare.csv')}")
print("[DONE]")
