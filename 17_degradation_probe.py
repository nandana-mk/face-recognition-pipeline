"""
17_degradation_probe.py — isolate EACH degradation and watch cosine similarity move.

Takes a set of CLEAN, large, near-frontal DroneFace crops (one identity's best shots), and for each
degrades it along ONE axis at a time, re-embeds, and measures cos(degraded, that identity's reference).
Because identity + pose are held fixed and only one factor varies, the resulting curve is the pure,
causal effect of that factor on the embedding.

Axes:
  RESOLUTION  downscale to k px then back to 112   (this is what "farther" really is)
  DARKNESS    scale pixel brightness down          (low light / dusk)
  BLUR        Gaussian blur (motion / focus)
  NOISE       additive Gaussian sensor noise
  JPEG        compression quality down             (bandwidth / codec)

Output: one overlaid curve plot (sim vs normalised severity) + one absolute-units plot per axis + CSV.
The gap between the curves IS the answer to "what breaks recognition": the steep ones destroy identity,
the flat ones (expected: darkness) barely touch it — those fail at DETECTION instead, not recognition.
"""
import os
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, csv, datetime
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
N_SOURCES_PER_SUBJECT = 2       # take each subject's 2 largest crops as clean sources
MIN_SOURCE_PX = 150             # only use genuinely large, detailed faces as the "clean" baseline

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

# ---------- references ----------
subjects = sorted({f.split('_')[0] for f in os.listdir(FACES) if f.lower().endswith(IMG_EXT)})
ref = {}
for s in subjects:
    fp = find_frontal(s)
    if fp:
        c = align_from_photo(fp)
        if c is not None: ref[s] = unit(c)
print(f"[INFO] {len(ref)} references")

# ---------- pick clean, large source crops ----------
by_subj = {}
for f in os.listdir(FACES):
    if not f.lower().endswith(IMG_EXT): continue
    p = f.split('_')
    if len(p) < 5 or p[1] != 'gp': continue
    sid = p[0]
    if sid not in ref: continue
    im = cv2.imread(os.path.join(FACES, f))
    if im is None: continue
    px = min(im.shape[0], im.shape[1])
    if px < MIN_SOURCE_PX: continue
    by_subj.setdefault(sid, []).append((px, im))
sources = []      # (sid, img112_clean)
for sid, lst in by_subj.items():
    lst.sort(key=lambda t: -t[0])
    for _, im in lst[:N_SOURCES_PER_SUBJECT]:
        sources.append((sid, cv2.resize(im, (112, 112), interpolation=cv2.INTER_AREA)))
print(f"[INFO] {len(sources)} clean source crops (>= {MIN_SOURCE_PX}px), {N_SOURCES_PER_SUBJECT}/subject")

# ---------- degradation functions ----------
def deg_resolution(img, k):        # downscale to k px then back to 112  (simulates distance)
    small = cv2.resize(img, (max(4, k), max(4, k)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (112, 112), interpolation=cv2.INTER_LINEAR)
def deg_dark(img, factor):         # brightness scale
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
def deg_blur(img, ks):
    ks = int(ks) | 1
    return cv2.GaussianBlur(img, (ks, ks), 0)
def deg_noise(img, sigma):
    n = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
def deg_jpeg(img, q):
    ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img

# (label, function, severity values ordered CLEAN->WORST, axis-unit text)
AXES = [
    ('resolution (px)', deg_resolution, [112, 96, 80, 64, 48, 40, 32, 24, 16], 'effective px'),
    ('darkness (bright x)', deg_dark, [1.0, 0.8, 0.6, 0.45, 0.3, 0.2, 0.12, 0.07], 'brightness factor'),
    ('blur (kernel)', deg_blur, [1, 3, 5, 7, 9, 13, 17, 23], 'gaussian kernel'),
    ('noise (sigma)', deg_noise, [0, 5, 10, 18, 28, 40, 55, 75], 'noise sigma'),
    ('jpeg (quality)', deg_jpeg, [100, 80, 60, 40, 25, 15, 8, 4], 'jpeg quality'),
]

np.random.seed(0)     # deterministic noise
rows = []
curves = {}           # label -> (severity_values, mean_sims)
for label, fn, sev, unit_txt in AXES:
    means = []
    for sv in sev:
        sims = []
        for sid, clean in sources:
            e = unit(fn(clean, sv))
            sims.append(float(e @ ref[sid]))
            rows.append({'axis': label, 'severity': sv, 'subject': sid, 'sim': round(sims[-1], 4)})
        means.append(float(np.mean(sims)))
    curves[label] = (sev, means)
    print(f"  {label:22s} clean={means[0]:.3f}  worst={means[-1]:.3f}  drop={means[0]-means[-1]:+.3f}")

# ---------- plot: normalised severity overlay ----------
pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("degrade_%H%M%S"))
os.makedirs(pdir, exist_ok=True)
COLORS = {'resolution (px)': '#C0392B', 'darkness (bright x)': '#2c3e50', 'blur (kernel)': '#8e44ad',
          'noise (sigma)': '#B67611', 'jpeg (quality)': '#0C7C88'}
fig, ax = plt.subplots(figsize=(10, 6))
for label, (sev, means) in curves.items():
    xs = np.linspace(0, 1, len(means))       # 0 = clean, 1 = worst
    ax.plot(xs, means, 'o-', lw=2.2, color=COLORS[label], label=label)
ax.axhline(THRESHOLD, color='grey', ls=':', lw=1.5, label=f'accept threshold {THRESHOLD}')
ax.set_xlabel('degradation severity (0 = clean  ->  1 = worst)'); ax.set_ylabel('cosine similarity to own reference')
ax.set_ylim(0, 1); ax.set_title('Which degradation destroys identity fastest?\n(steep = kills recognition; flat = recognition survives, fails only at detection)')
ax.grid(alpha=.3); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(pdir, 'degradation_overlay.png'), dpi=150); plt.close(fig)

# ---------- plot: each axis in its real units ----------
fig, axs = plt.subplots(2, 3, figsize=(16, 9))
for a, (label, (sev, means)) in zip(axs.flat, curves.items()):
    a.plot(sev, means, 'o-', lw=2.2, color=COLORS[label])
    a.axhline(THRESHOLD, color='grey', ls=':')
    a.set_title(label); a.set_ylabel('cosine sim'); a.set_ylim(0, 1); a.grid(alpha=.3)
    if 'resolution' in label or 'darkness' in label or 'jpeg' in label: a.invert_xaxis()  # worst on the right
axs.flat[-1].axis('off')
fig.suptitle('Cosine similarity vs each degradation in real units (dotted = accept threshold)', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, .96]); fig.savefig(os.path.join(pdir, 'degradation_per_axis.png'), dpi=150); plt.close(fig)

with open(os.path.join(OUT_DIR, "degradation_probe.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

print(f"\n[SAVED] 2 plots -> {pdir}")
print(f"[SAVED] CSV -> {os.path.join(OUT_DIR, 'degradation_probe.csv')}")
print("\nExpected reading: resolution and noise curves plunge through the threshold; darkness stays")
print("high and flat (ArcFace is brightness-robust) — confirming darkness breaks DETECTION, not identity.")
print("[DONE]")
