"""
32_large_validation.py — large-scale validation + threshold-vs-gallery-size law.

    python 32_large_validation.py "C:\\dev\\face-recognition-pipeline\\SCface_database" ^
                                  "C:\\Users\\hticd\\scikit_learn_data\\lfw_home\\lfw_funneled"

THE CONSTRAINT THIS RESPECTS
    Your watchlist faces (Nandana, Siddhartha, Allen, Nitharshan, Abhinav, Dakshina, Sakthi)
    are TEST-ONLY. Nothing here fits anything on them. The operating threshold is derived
    entirely from SCface identities and LFW strangers, then APPLIED to your data once, as a
    genuine held-out test (PART 4).

WHY A GALLERY-SIZE SWEEP
    A threshold does not transfer between deployments - the required value rises with the number
    of enrolled people, because max-over-gallery is a larger order statistic when the gallery is
    bigger. Every validation so far used 11, 98 or 130 identities; YOUR watchlist is ~9. So we
    measure the LAW (threshold vs gallery size) on surrogate identities and read off the value
    for a 9-person gallery.

WHY THE STRANGER SET IS BIG
    450-1500 strangers cannot resolve false-accept rates below ~0.2%; every "0% FA" threshold we
    computed was pinned to a single worst stranger. Here we use SCface's non-enrolled subjects
    (same cameras, realistic near-domain) PLUS thousands of degraded LFW faces (scale), and
    report both separately - they measure different things.

PARTS
    1  embed everything once (SCface gallery + probes, LFW strangers)
    2  gallery-size sweep with k-fold folds -> threshold and recall, mean +- std
    3  the threshold-vs-gallery-size law, and the predicted value for YOUR gallery size
    4  apply that predicted threshold to YOUR data (held-out test, nothing fitted on it)
"""
import os, re, sys, glob, csv
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, datetime
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.utils import face_align

OUT_DIR    = r"C:\dev\train_val\reports"
LABELED    = r"C:\dev\train_val\reports\labeled_eval.csv"   # YOUR data, PART 4 only
KNOWN_DIR  = r"C:\dev\face-recognition-pipeline\Known_faces"
N_LFW      = 6000              # degraded LFW strangers -> FAR resolution ~0.017%
GALLERY_SIZES = [5, 9, 15, 25, 50, 100]
FOLDS      = 5
TARGET_PRECISION = 1.0
YOUR_GALLERY_SIZE = 9
COS_GRID   = np.round(np.arange(0.10, 0.701, 0.005), 3)
IMG_EXT    = ('.jpg', '.jpeg', '.png', '.bmp')
SEED       = 21

rng = np.random.default_rng(SEED)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def emb(c):
    f = rec.get_feat(c); f = np.asarray(f, np.float32).reshape(len(c), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)
def emb_all(c, bs=256):
    return np.concatenate([emb(c[i:i+bs]) for i in range(0, len(c), bs)], 0) if c else np.zeros((0, 512), np.float32)

def align_largest(img):
    bb, kps = det.detect(img, max_num=0, metric='default')
    if bb is not None and len(bb):
        j = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        return face_align.norm_crop(img, kps[j]), kps[j], float(min(bb[j, 2]-bb[j, 0], bb[j, 3]-bb[j, 1]))
    h, w = img.shape[:2]; pad = int(0.4*max(h, w))
    p = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    bb, kps = det.detect(p, max_num=0, metric='default')
    if bb is not None and len(bb):
        j = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        return face_align.norm_crop(p, kps[j]), kps[j], float(min(bb[j, 2]-bb[j, 0], bb[j, 3]-bb[j, 1]))
    if h and w and 0.6 <= w/h <= 1.8 and min(h, w) >= 60:
        return cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA), None, float(min(h, w))
    return None, None, 0.0

def degrade(im, kps, w, r):
    """realistic: shrink the whole image so the face is k px, corrupt, then RE-DETECT"""
    k = int(r.integers(25, 75)); s = k / max(w, 1e-6)
    H, W = im.shape[:2]
    sm = cv2.resize(im, (max(16, int(W*s)), max(16, int(H*s))), interpolation=cv2.INTER_AREA)
    if r.random() < 0.8:
        sm = np.clip(sm.astype(np.float32) + r.normal(0, r.uniform(3, 14), sm.shape), 0, 255).astype(np.uint8)
    bb2, k2 = det.detect(sm, max_num=0, metric='default')
    if bb2 is not None and len(bb2):
        j = int(np.argmax((bb2[:, 2]-bb2[:, 0])*(bb2[:, 3]-bb2[:, 1])))
        return face_align.norm_crop(sm, k2[j])
    if kps is None: return None
    return face_align.norm_crop(sm, (kps*s + r.normal(0, 1.0, kps.shape)).astype(np.float32))

# ============================== PART 1: embed once ==============================
sc_root = sys.argv[1] if len(sys.argv) > 1 else None
lfw_root = sys.argv[2] if len(sys.argv) > 2 else None
if not sc_root or not os.path.isdir(sc_root):
    print("usage: python 32_large_validation.py <SCface root> <lfw folder>"); raise SystemExit

files = [p for p in glob.glob(os.path.join(sc_root, '**', '*'), recursive=True)
         if p.lower().endswith(IMG_EXT)]
files.sort(key=lambda p: (0 if 'cropped' in p.lower() else 1,
                          0 if 'surveillance_cameras_all' in p.replace('\\', '/') else 1, p))
re_s = re.compile(r'(\d{3})_cam(\d+)_(\d)', re.I); re_m = re.compile(r'(\d{3})[_\-]?(frontal|mugshot)', re.I)
mug, surv, seen = {}, [], set()
for p in files:
    b, low = os.path.basename(p), p.lower()
    if 'rotation' in low: continue
    m = re_m.search(b)
    if m and ('mug' in low or 'frontal' in b.lower()): mug.setdefault(m.group(1), p); continue
    s = re_s.search(b)
    if s and int(s.group(2)) <= 5 and b.lower() not in seen:
        seen.add(b.lower()); surv.append((s.group(1), int(s.group(2)), s.group(3), p))

gal_ids, gal_crops = [], []
for sid in sorted(mug):
    im = cv2.imread(mug[sid])
    if im is None: continue
    c, _, _ = align_largest(im)
    if c is not None: gal_ids.append(sid); gal_crops.append(c)
GAL = emb_all(gal_crops); gal_ids = np.array(gal_ids)

pc, pid, pdist, ppx = [], [], [], []
for sid, cam, dc, p in surv:
    if sid not in gal_ids: continue
    im = cv2.imread(p)
    if im is None: continue
    c, _, px = align_largest(im)
    if c is None: continue
    pc.append(c); pid.append(sid); pdist.append(dc); ppx.append(px)
PROBE = emb_all(pc); pid = np.array(pid); pdist = np.array(pdist); ppx = np.array(ppx)
print(f"[SCface] gallery {len(gal_ids)} identities | probes {len(pc)}")

LFW = np.zeros((0, 512), np.float32)
if lfw_root and os.path.isdir(lfw_root):
    lf = [p for p in glob.glob(os.path.join(lfw_root, '**', '*'), recursive=True)
          if p.lower().endswith(IMG_EXT)]
    rng.shuffle(lf)
    sc_ = []
    for p in lf:
        if len(sc_) >= N_LFW: break
        im = cv2.imread(p)
        if im is None: continue
        _, kps, w = align_largest(im)
        if w < 50: continue
        d = degrade(im, kps, w, rng)
        if d is not None: sc_.append(d)
        if len(sc_) % 1000 == 0 and len(sc_): print(f"  ...LFW strangers {len(sc_)}/{N_LFW}")
    LFW = emb_all(sc_)
print(f"[LFW] {LFW.shape[0]} degraded strangers "
      f"(FAR resolution ~{100.0/max(LFW.shape[0],1):.3f}%)\n")

# ============================== PART 2: gallery-size sweep ==============================
S_all = PROBE @ GAL.T           # (probes, 130)
L_all = LFW @ GAL.T if LFW.shape[0] else np.zeros((0, len(gal_ids)), np.float32)

def eval_fold(idx_enrolled, thr_grid):
    """returns per-threshold (precision, recall) for this enrolled subset"""
    names = gal_ids[idx_enrolled]
    S = S_all[:, idx_enrolled]
    is_gen = np.isin(pid, names)
    top = names[S.argmax(1)]; best = S.max(1)
    correct = (top == pid) & is_gen
    L = L_all[:, idx_enrolled].max(1) if L_all.shape[0] else np.zeros(0)
    out = []
    for t in thr_grid:
        acc = best >= t
        tp = int(np.sum(acc & correct))
        wrong = int(np.sum(acc & is_gen & ~correct))
        fa_sc = int(np.sum(acc & ~is_gen))          # SCface non-enrolled = near-domain strangers
        fa_lf = int(np.sum(L >= t)) if L.size else 0
        prec = tp / max(tp + wrong + fa_sc + fa_lf, 1)
        rec = tp / max(int(is_gen.sum()), 1)
        out.append((prec, rec, wrong, fa_sc, fa_lf))
    return out

print("=" * 96)
print("PART 2 — GALLERY-SIZE SWEEP  (threshold needed to hold precision >= "
      f"{TARGET_PRECISION:.3f}, {FOLDS} folds)")
print("=" * 96)
print(f"  {'gallery':>8} {'thr mean':>9} {'thr std':>8} {'recall mean':>12} {'recall std':>11} "
      f"{'strangers':>10}")
law = []
for gs in GALLERY_SIZES:
    if gs > len(gal_ids): continue
    thrs, recs = [], []
    for f in range(FOLDS):
        idx = rng.choice(len(gal_ids), size=gs, replace=False)
        res = eval_fold(idx, COS_GRID)
        ok = [(COS_GRID[i], r[1]) for i, r in enumerate(res) if r[0] >= TARGET_PRECISION]
        if not ok: continue
        t, r = max(ok, key=lambda e: e[1])
        thrs.append(t); recs.append(r)
    if not thrs: continue
    n_str = int((~np.isin(pid, gal_ids[:gs])).sum()) + LFW.shape[0]
    law.append((gs, float(np.mean(thrs)), float(np.std(thrs)), float(np.mean(recs))))
    print(f"  {gs:>8} {np.mean(thrs):>9.3f} {np.std(thrs):>8.3f} {np.mean(recs):>11.1%} "
          f"{np.std(recs):>10.1%} {n_str:>10}")

# ============================== PART 3: the law ==============================
print("\n" + "=" * 96)
print("PART 3 — THRESHOLD vs GALLERY SIZE")
print("=" * 96)
if len(law) >= 2:
    g = np.array([l[0] for l in law], float); t = np.array([l[1] for l in law])
    a, b = np.polyfit(np.log(g), t, 1)
    pred = a*np.log(YOUR_GALLERY_SIZE) + b
    print(f"  fit:  threshold = {a:.4f} * ln(gallery_size) + {b:.4f}")
    for gs, tm, ts, rm in law:
        print(f"    {gs:>4} identities -> {tm:.3f}  (fit {a*np.log(gs)+b:.3f})")
    print(f"\n  >>> PREDICTED threshold for YOUR {YOUR_GALLERY_SIZE}-person watchlist: "
          f"{pred:.3f}")
    if a > 0.005:
        print(f"      (threshold rises with gallery size, as expected: +{a:.3f} per e-fold)")
    else:
        print("      (little gallery-size dependence here - a single threshold may transfer)")
else:
    pred = None; print("  not enough points to fit")

# ============================== PART 4: YOUR data, held out ==============================
print("\n" + "=" * 96)
print("PART 4 — HELD-OUT TEST ON YOUR DATA  (nothing above was fitted on these faces)")
print("=" * 96)
if pred is None or not os.path.exists(LABELED):
    print(f"  skipped (need {LABELED} and a fitted law)")
else:
    names_y, vecs_y = [], []
    for e in sorted(os.listdir(KNOWN_DIR)):
        full = os.path.join(KNOWN_DIR, e)
        if os.path.isdir(full) and not e.startswith('_'):
            person = e; fs = [os.path.join(full, x) for x in sorted(os.listdir(full))
                              if x.lower().endswith(IMG_EXT)]
        elif e.lower().endswith(IMG_EXT):
            person, fs = os.path.splitext(e)[0], [full]
        else:
            continue
        vv = []
        for p in fs:
            im = cv2.imread(p)
            if im is None: continue
            c, _, _ = align_largest(im)
            if c is not None: vv.append(emb([c])[0])
        if vv:
            m = np.mean(vv, 0); names_y.append(person); vecs_y.append(m/(np.linalg.norm(m)+1e-9))
    GY = np.array(vecs_y, np.float32); names_y = np.array(names_y)
    print(f"  your gallery: {len(names_y)} identities ({', '.join(names_y)})")

    rows = list(csv.DictReader(open(LABELED, newline='', encoding='utf-8')))

    # The 'impostor' column in labeled_eval.csv was computed against the OLD video_dataset
    # gallery, which has DIFFERENT membership and DIFFERENT spellings from Known_faces
    # (Abhinav vs Abhinav1, Harsh vs Harshana, Dakshina enrolled there but not here, Jaga the
    # reverse). Using it would score against the wrong ground truth. So: resolve each label to a
    # gallery identity by case-insensitive prefix match, and define genuine/impostor by ACTUAL
    # gallery membership.
    import difflib
    lower = {n.lower(): n for n in names_y}
    def resolve(label):
        l = label.strip().lower()
        if l in lower: return lower[l]
        hits = [n for n in names_y if n.lower().startswith(l) or l.startswith(n.lower())]
        if len(hits) == 1: return hits[0]
        # near-miss spellings, e.g. label 'Tejewar' vs gallery 'Tejeswar' (prefix match fails
        # because they diverge mid-word). Without this, 33 crops of an ENROLLED person were
        # scored as impostors, and every correct identification of him as a false accept.
        close = difflib.get_close_matches(l, list(lower.keys()), n=1, cutoff=0.85)
        return lower[close[0]] if close else None

    labels = sorted({r['true'] for r in rows})
    mapping = {lb: resolve(lb) for lb in labels}
    print("  label -> gallery resolution (None = not on the watchlist = impostor):")
    for lb in labels:
        n = sum(1 for r in rows if r['true'] == lb)
        print(f"    {lb:<14} -> {str(mapping[lb]):<14} ({n} crops)")

    cy, ty, iy = [], [], []
    for r in rows:
        im = cv2.imread(r['path'])
        if im is None: continue
        c = cv2.resize(im, (112, 112)) if im.shape[:2] != (112, 112) else im
        cy.append(cv2.bilateralFilter(c, 5, 50, 50))
        m = mapping.get(r['true'])
        ty.append(m if m else r['true'])
        iy.append(m is None)                     # impostor = not resolvable to the gallery
    PY = emb_all(cy); ty = np.array(ty); iy = np.array(iy, bool)
    print(f"\n  after resolution: {int((~iy).sum())} genuine, {int(iy.sum())} impostor crops")
    SY = PY @ GY.T
    topy = names_y[SY.argmax(1)]; besty = SY.max(1)
    for label, thr in [("PREDICTED from the law", pred), ("your current 0.40", 0.40)]:
        acc = besty >= thr
        tp = int(np.sum(acc & ~iy & (topy == ty)))
        wrong = int(np.sum(acc & ~iy & (topy != ty)))
        fa = int(np.sum(acc & iy))
        prec = tp/max(tp+wrong+fa, 1); rec = tp/max(int((~iy).sum()), 1)
        print(f"  {label:<26} thr {thr:.3f}  precision {prec:.4f}  recall {rec:.1%}  "
              f"F1 {2*prec*rec/max(prec+rec,1e-9):.4f}  (wrong-name {wrong}, stranger-accept {fa})")

# ---------------- plot ----------------
if law:
    pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("largeval_%H%M%S"))
    os.makedirs(pdir, exist_ok=True)
    g = [l[0] for l in law]; t = [l[1] for l in law]; e = [l[2] for l in law]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(g, t, yerr=e, fmt='o-', color='#0C7C88', capsize=4, label='measured')
    if pred is not None:
        ax.axvline(YOUR_GALLERY_SIZE, ls=':', color='#C0392B')
        ax.plot([YOUR_GALLERY_SIZE], [pred], '*', ms=16, color='#C0392B',
                label=f'your watchlist ({YOUR_GALLERY_SIZE}) -> {pred:.3f}')
    ax.set_xscale('log'); ax.set_xlabel('gallery size (identities)')
    ax.set_ylabel(f'cos_threshold for precision >= {TARGET_PRECISION}')
    ax.set_title('Threshold required vs gallery size'); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(pdir, 'threshold_vs_gallery.png'), dpi=150)
    print(f"\n[SAVED] plot -> {pdir}")
