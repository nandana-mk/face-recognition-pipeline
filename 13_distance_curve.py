"""
13_distance_curve.py — controlled recognition-vs-distance benchmark.
Runs the fr2 pipeline (Known_faces + denoise + far-detect + temporal aggregation) on the
per-meter clips (2m.MP4 ... 8m.MP4) and reports, per exact metre: recognized people, their
similarity, face px, and whether both expected people were found. Saves a CSV + a plot.
FILL `EXPECTED` with the two people standing in these clips (must be enrolled in Known_faces).
"""
import os
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, csv, re, datetime
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis
from insightface.utils import face_align

VIDEO_DIR = r"C:\Users\hticd\OneDrive\Desktop\DJI data\DJI Avata & Mini 4K\2m-8m"
KNOWN_DIR = r"C:\dev\face-recognition-pipeline\Known_faces"
OUT_DIR   = r"C:\dev\train_val\reports"
EXPECTED  = set()          # <-- FILL: the 2 people in these clips, e.g. {"Nandana", "Siddhartha"}

THRESHOLD = 0.40
FRAME_SKIP = 3
DET_SIZE = 1280
SCALES = [0.5, 1.0, 2.0]
MIN_FACE = 12
MIN_DET = 0.15
IOU_T = 0.2
MIN_FRAMES = 5
EXT = ('.mp4', '.mov')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
det, rec = app.models['detection'], app.models['recognition']

def embed(face112):
    face112 = cv2.bilateralFilter(face112, 5, 50, 50)
    e = rec.get_feat(face112).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
def enhance_det(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    img = cv2.cvtColor(cv2.merge([_clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    m = img.mean()
    if m < 110:
        g = 1.6 if m < 70 else 1.3
        lut = np.array([((i/255.0)**(1.0/g))*255 for i in range(256)]).astype('uint8')
        img = cv2.LUT(img, lut)
    return img

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

# enroll (single Known_faces photo per person)
names, gal = [], []
for f in sorted(os.listdir(KNOWN_DIR)):
    if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')): continue
    im = cv2.imread(os.path.join(KNOWN_DIR, f))
    if im is None: continue
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0: continue
    k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
    names.append(os.path.splitext(f)[0]); gal.append(embed(face_align.norm_crop(im, kps[k])))
G = np.array(gal, np.float32)
print(f"[INFO] enrolled: {', '.join(names)}")
if not EXPECTED:
    print("[WARN] EXPECTED is empty — set the 2 people in the clips to get recall/curve scoring.\n")

def meter_of(fn):
    m = re.search(r'(\d+)\s*m', fn.lower())
    return int(m.group(1)) if m else None

def match(v):
    s = G @ v; i = int(np.argmax(s)); return names[i], float(s[i])

clips = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(EXT)],
               key=lambda f: (meter_of(f) if meter_of(f) is not None else 999))

rows = []
for clip in clips:
    meter = meter_of(clip)
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, clip))
    if not cap.isOpened(): print(f"[skip] {clip}"); continue
    tracks = {}; nid = 0; fnum = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        fnum += 1
        if fnum % FRAME_SKIP: continue
        det_in = enhance_det(frame); faces = []
        for s in SCALES:
            sc = det_in if s == 1.0 else cv2.resize(det_in, None, fx=s, fy=s)
            bb, kps = det.detect(sc, max_num=0, metric='default')
            if bb is None: continue
            for k in range(len(bb)):
                if bb[k, 4] < MIN_DET: continue
                faces.append((bb[k, :4]/s, kps[k]/s))
        for box, kp in faces:
            x1, y1, x2, y2 = [int(v) for v in box]
            px = min(x2-x1, y2-y1)
            if px < MIN_FACE: continue
            e = embed(face_align.norm_crop(frame, kp))
            bi, bt = 0, None
            for tid, t in tracks.items():
                v = iou((x1, y1, x2, y2), t['bbox'])
                if v > bi and v > IOU_T: bi, bt = v, tid
            if bt is None:
                bt = nid; tracks[nid] = {'bbox': (x1, y1, x2, y2), 'sum': e.copy(), 'n': 1, 'pxs': [px]}; nid += 1
            else:
                t = tracks[bt]; t['bbox'] = (x1, y1, x2, y2); t['sum'] += e; t['n'] += 1; t['pxs'].append(px)
    cap.release()
    recognized = []
    for tid, t in tracks.items():
        if t['n'] < MIN_FRAMES: continue
        agg = t['sum'] / (np.linalg.norm(t['sum']) + 1e-9)
        nm, sim = match(agg)
        decision = nm if sim >= THRESHOLD else "Unknown"
        rows.append({'meter': meter, 'clip': clip, 'track': tid, 'pred': nm, 'sim': round(sim, 3),
                     'px_min': min(t['pxs']), 'px_max': max(t['pxs']), 'decision': decision})
        recognized.append((decision, sim, max(t['pxs'])))
    tag = f"{meter}m" if meter is not None else clip
    recs = ", ".join(f"{d}({s:.2f},{p}px)" for d, s, p in sorted(recognized, key=lambda x: -x[2]))
    print(f"  {tag:>6}: {recs or 'no faces'}")

# ---------- curve / scoring ----------
print("\n" + "=" * 66)
print("RECOGNITION vs DISTANCE")
print("=" * 66)
meters = sorted(set(r['meter'] for r in rows if r['meter'] is not None))
curve = []
print(f"  {'metre':>5} {'expected-found':>16} {'mean-sim(found)':>16}")
for mtr in meters:
    mr = [r for r in rows if r['meter'] == mtr]
    found = {r['pred'] for r in mr if r['decision'] != 'Unknown' and (not EXPECTED or r['pred'] in EXPECTED)}
    exp_found = found & EXPECTED if EXPECTED else found
    sims = [r['sim'] for r in mr if r['pred'] in exp_found]
    rate = f"{len(exp_found)}/{len(EXPECTED)}" if EXPECTED else f"{len(found)}"
    ms = np.mean(sims) if sims else 0.0
    curve.append((mtr, len(exp_found), ms))
    print(f"  {mtr:>4}m {rate:>16} {ms:>16.3f}")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "distance_curve.csv"), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

if curve:
    pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("dist_%H%M%S"))
    os.makedirs(pdir, exist_ok=True)
    ms_ = [c[0] for c in curve]
    plt.figure(figsize=(8, 5))
    plt.plot(ms_, [c[2] for c in curve], 'o-', color='steelblue', lw=2)
    plt.axhline(THRESHOLD, ls='--', color='gray', label=f'threshold {THRESHOLD}')
    plt.xlabel('distance (metres)'); plt.ylabel('mean recognition similarity'); plt.ylim(0, 1)
    plt.title('Recognition similarity vs distance'); plt.grid(alpha=0.3); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(pdir, 'sim_vs_distance.png'), dpi=150); plt.close()
    print(f"\n[SAVED] distance_curve.csv + plot -> {pdir}")
print("[DONE]")
