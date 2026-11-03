"""
16_embedding_forensics.py — LOOK INSIDE the cosine similarity on real DroneFace crops.

For every crop it recovers the full 512-d embedding and decomposes the decision into its parts:
  genuine_sim   = cos(crop, the crop's OWN enrolled reference)      -> identity signal that survives
  impostor_sim  = max cos(crop, every OTHER reference)              -> how close the nearest wrong person is
  margin        = genuine_sim - impostor_sim                        -> the real headroom (this is what breaks)
  feat_norm     = ||raw embedding|| BEFORE normalization            -> ArcFace's built-in quality/confidence signal
  sim_to_mean   = cos(crop, mean of all references)                 -> pull toward the generic "average face"

Then it plots WHY recognition dies with distance:
  A  genuine vs impostor similarity vs face-pixel-size (+ margin shading, + threshold)   <- the money plot
  B  margin vs distance, by camera height (shows pitch eating the margin)
  C  genuine vs impostor DISTRIBUTIONS at NEAR (px>=90) vs FAR (px<50) — watch them merge
  D  feature norm vs px (embedding "confidence" collapses before the cosine does)
  E  genuine_sim vs sim_to_mean vs px (identity signal bleeding into the generic face)
  F  2-D PCA of all embeddings, coloured by identity, faded by distance (clusters dissolving)

No fine-tuning, no denoise (we want the RAW signal). Outputs a timestamped plot folder + a per-crop CSV.
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
HEIGHT_MAP = {'0': 1.5, '3': 3.0, '4': 4.0, '5': 5.0}

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

def raw_feat(img):
    """return (unnormalised 512-d, its L2 norm)."""
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)
    f = rec.get_feat(img).flatten().astype(np.float32)
    return f, float(np.linalg.norm(f) + 1e-9)

def unit(img):
    f, n = raw_feat(img); return f / n

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

# ---------- references ----------
subjects = sorted({f.split('_')[0] for f in os.listdir(FACES) if f.lower().endswith(IMG_EXT)})
names, refs = [], []
for s in subjects:
    fp = find_frontal(s)
    if not fp: print(f"  [WARN] no portrait for {s}"); continue
    crop = align_from_photo(fp)
    if crop is None: continue
    names.append(s); refs.append(unit(crop))
R = np.array(refs, np.float32)            # (K,512) unit reference vectors
Rmean = R.mean(0); Rmean /= (np.linalg.norm(Rmean) + 1e-9)   # the "generic face" direction
idx = {n: i for i, n in enumerate(names)}
print(f"[INFO] enrolled {len(names)} references: {names}")

# average reference-to-reference similarity (baseline 'different-person' level)
RR = R @ R.T; np.fill_diagonal(RR, np.nan)
print(f"[INFO] mean reference<->reference (different people) cosine = {np.nanmean(RR):.3f}")

# ---------- embed every crop ----------
rows, embs, emb_sid, emb_px = [], [], [], []
for f in os.listdir(FACES):
    if not f.lower().endswith(IMG_EXT): continue
    p = f.split('_')
    if len(p) < 5 or p[1] != 'gp': continue
    sid, gp, seq = p[0], p[2], p[4].split('.')[0]
    if sid not in idx: continue
    im = cv2.imread(os.path.join(FACES, f))
    if im is None: continue
    px = min(im.shape[0], im.shape[1])
    fr, nrm = raw_feat(im); e = fr / nrm
    sims = R @ e
    gi = idx[sid]
    gen = float(sims[gi])
    others = sims.copy(); others[gi] = -1
    ii = int(np.argmax(others)); imp = float(others[ii])
    rows.append({'subject': sid, 'dist_m': 17 - int(seq)/2.0, 'height_m': HEIGHT_MAP.get(gp, gp),
                 'px': px, 'feat_norm': round(nrm, 2), 'genuine_sim': round(gen, 4),
                 'impostor_sim': round(imp, 4), 'impostor_who': names[ii],
                 'margin': round(gen - imp, 4), 'sim_to_mean': round(float(e @ Rmean), 4)})
    embs.append(e); emb_sid.append(sid); emb_px.append(px)
embs = np.array(embs, np.float32)
print(f"[INFO] embedded {len(rows)} crops")

# ---------- helpers ----------
PXBINS = [(0,35),(35,45),(45,55),(55,70),(70,90),(90,120),(120,10000)]
PXCENT = [30,40,50,62,80,105,150]
def binned(metric):
    ys = []
    for lo, hi in PXBINS:
        v = [r[metric] for r in rows if lo <= r['px'] < hi]
        ys.append(np.mean(v) if v else np.nan)
    return ys

pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("forensics_%H%M%S"))
os.makedirs(pdir, exist_ok=True)
TEAL, RED, AMBER, GREY, PURPLE = '#0C7C88', '#C0392B', '#B67611', '#7f8c8d', '#8e44ad'

# ---------- A: genuine vs impostor vs px (THE money plot) ----------
g, imp = binned('genuine_sim'), binned('impostor_sim')
fig, ax = plt.subplots(figsize=(9.5, 5.5))
ax.plot(PXCENT, g, 'o-', color=TEAL, lw=2.4, label='genuine sim (own reference)')
ax.plot(PXCENT, imp, 's-', color=RED, lw=2.0, label='impostor sim (nearest wrong person)')
ax.fill_between(PXCENT, g, imp, where=[a >= b for a, b in zip(g, imp)], color=TEAL, alpha=.12, label='margin (headroom)')
ax.axhline(THRESHOLD, color=GREY, ls=':', lw=1.5, label=f'accept threshold {THRESHOLD}')
ax.axhline(float(np.nanmean(RR)), color=RED, ls='--', alpha=.5, label='baseline different-person level')
ax.set_xlabel('face size (px)'); ax.set_ylabel('cosine similarity'); ax.set_ylim(0, 1)
ax.set_title('A · What breaks: genuine sim decays to meet the impostor floor')
ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'A_genuine_vs_impostor.png'), dpi=150); plt.close(fig)

# ---------- B: margin vs distance, by height ----------
fig, ax = plt.subplots(figsize=(9, 5))
cmap = plt.cm.viridis(np.linspace(0, .85, 4))
for h, col in zip([1.5, 3.0, 4.0, 5.0], cmap):
    dd = defaultdict(list)
    for r in rows:
        if r['height_m'] == h: dd[r['dist_m']].append(r['margin'])
    xs = sorted(dd); ax.plot(xs, [np.mean(dd[x]) for x in xs], '-', color=col, lw=1.8, label=f'{h:g} m')
ax.axhline(0, color=RED, ls='--', alpha=.6, label='margin = 0 (identity lost)')
ax.set_xlabel('distance (m)'); ax.set_ylabel('genuine − impostor margin')
ax.set_title('B · The margin vs distance, by camera height (pitch eats the margin)')
ax.grid(alpha=.3); ax.legend(title='height', fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'B_margin_vs_distance.png'), dpi=150); plt.close(fig)

# ---------- C: distributions NEAR vs FAR ----------
near = [r for r in rows if r['px'] >= 90]; far = [r for r in rows if r['px'] < 50]
fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
bins = np.linspace(-0.1, 1.0, 45)
for a, grp, ttl in [(ax[0], near, f'NEAR  (px>=90, n={len(near)})'), (ax[1], far, f'FAR  (px<50, n={len(far)})')]:
    a.hist([r['genuine_sim'] for r in grp], bins=bins, color=TEAL, alpha=.6, label='genuine')
    a.hist([r['impostor_sim'] for r in grp], bins=bins, color=RED, alpha=.55, label='impostor')
    a.axvline(THRESHOLD, color=GREY, ls=':', label='threshold')
    a.set_title(ttl); a.set_xlabel('cosine similarity'); a.grid(alpha=.3); a.legend(fontsize=8)
ax[0].set_ylabel('crops')
fig.suptitle('C · Genuine (blue) and impostor (red) distributions — separated up close, merged far away', fontsize=12)
fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(os.path.join(pdir, 'C_distributions_near_far.png'), dpi=150); plt.close(fig)

# ---------- D: feature norm vs px ----------
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.scatter([r['px'] for r in rows], [r['feat_norm'] for r in rows], s=6, alpha=.25, color=PURPLE)
fn = binned('feat_norm')
ax.plot(PXCENT, fn, 'o-', color='black', lw=2, label='mean')
ax.set_xlabel('face size (px)'); ax.set_ylabel('raw embedding norm  ||f||')
ax.set_title("D · ArcFace feature norm vs px — the model's own confidence collapses with resolution")
ax.set_xlim(0, 260); ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'D_feature_norm.png'), dpi=150); plt.close(fig)

# ---------- E: genuine vs sim-to-mean ----------
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(PXCENT, binned('genuine_sim'), 'o-', color=TEAL, lw=2.2, label='sim to OWN reference (identity)')
ax.plot(PXCENT, binned('sim_to_mean'), 's--', color=AMBER, lw=2.0, label='sim to AVERAGE face (generic)')
ax.set_xlabel('face size (px)'); ax.set_ylabel('cosine similarity')
ax.set_title('E · As faces shrink, identity signal fades and the crop drifts toward the generic face')
ax.grid(alpha=.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'E_identity_vs_generic.png'), dpi=150); plt.close(fig)

# ---------- F: PCA scatter ----------
X = embs - embs.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
C = X @ Vt[:2].T
pxarr = np.array(emb_px, float)
alpha = np.clip((pxarr - 30) / 120.0, 0.08, 0.9)     # near = opaque, far = faint
palette = plt.cm.tab20(np.linspace(0, 1, len(names)))
fig, ax = plt.subplots(figsize=(9, 8))
for i, n in enumerate(names):
    m = [j for j in range(len(emb_sid)) if emb_sid[j] == n]
    ax.scatter(C[m, 0], C[m, 1], s=16, color=palette[i], alpha=0.5, label=n, edgecolors='none')
# faint = far crops, drawn on top to show the central collapse
ax.set_title('F · Embedding map (PCA). Each colour = one identity.\nTight coloured clusters = near/large faces; the central blur = far faces collapsing together')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(ncol=2, fontsize=8, title='subject'); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'F_embedding_pca.png'), dpi=150); plt.close(fig)

# ---------- CSV ----------
with open(os.path.join(OUT_DIR, "embedding_forensics.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"\n[SAVED] 6 plots -> {pdir}")
print(f"[SAVED] per-crop CSV -> {os.path.join(OUT_DIR, 'embedding_forensics.csv')}")
print("\nRead them in order A->F: A shows the genuine line falling to meet the impostor floor;")
print("F shows the same thing geometrically — identities collapsing into one central blob at distance.")
print("[DONE]")
