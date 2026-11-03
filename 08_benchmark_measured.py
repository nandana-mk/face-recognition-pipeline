"""
08_benchmark_measured.py — watchlist benchmark on a drone video.
- Enrolls each target from ONE Known_faces photo (real watchlist), applying far_adapter.
- Runs the video with far-detection + per-TRACK temporal aggregation (average a person's
  embeddings over the frames they are visible -> one strong identity per person).
- Reports per-frame AND per-track results, ADAPTER vs RAW, + a distance (px) curve.
Usage:  python 08_benchmark_measured.py ["path\to\video.mp4"]
"""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2, numpy as np, json, sys, csv
import torch, torch.nn as nn, torch.nn.functional as F
from collections import defaultdict
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ---------- config ----------
VIDEO = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\hticd\OneDrive\Desktop\DJI data\DJI Avata & Mini 4K\measured distance.mp4"
KNOWN_DIR = r"C:\dev\face-recognition-pipeline\Known_faces"
MODEL_DIR = r"C:\dev\train_val\models"
CSV_OUT   = r"C:\dev\train_val\reports\measured_benchmark.csv"
# ONE reference photo per target -> display name (Kailash = Krishna Kailash)
KNOWN = {"Abhinav1.jpeg": "Abhinav", "Kailash.jpg": "Krishna", "Tejeswar.jpg": "Tejeswar"}

SIM_THRESHOLD = 0.38
FRAME_SKIP = 2
MIN_FACE_SIZE = 12
MIN_DET_SCORE = 0.15
SCALES = [0.5, 1.0, 2.0]
IOU_THRESHOLD = 0.2
DISPLAY_MAX_W = 1600
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- models ----------
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1280, 1280))
det_model = app.models['detection']; rec_model = app.models['recognition']

with open(os.path.join(MODEL_DIR, "far_class_map.json")) as fh:
    meta = json.load(fh)
class MetricAdapter(nn.Module):
    def __init__(self, i, o):
        super().__init__(); self.proj = nn.Sequential(nn.Linear(i, o, bias=False), nn.BatchNorm1d(o))
    def forward(self, x): return F.normalize(self.proj(x), dim=1)
adapter = MetricAdapter(meta['in_dim'], meta['embed_dim']).to(DEVICE)
adapter.load_state_dict(torch.load(os.path.join(MODEL_DIR, "far_adapter.pt"), map_location=DEVICE))
adapter.eval()
print(f"[INFO] far_adapter loaded ({meta['in_dim']}->{meta['embed_dim']}), device {DEVICE}")

def raw_emb(face112):
    e = rec_model.get_feat(face112).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)
def adapt(e):
    with torch.no_grad():
        v = adapter(torch.from_numpy(e[None]).float().to(DEVICE)).cpu().numpy()[0]
    return v

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

# ---------- enroll (one photo each) ----------
names, g_raw, g_adp = [], [], []
for fn, disp in KNOWN.items():
    p = os.path.join(KNOWN_DIR, fn)
    im = cv2.imread(p)
    if im is None:
        print(f"  [skip] {fn} not found"); continue
    fs = det_model.detect(im, max_num=0, metric='default')
    bb, kps = fs
    if bb is None or len(bb) == 0:
        print(f"  [skip] no face in {fn}"); continue
    k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
    r = raw_emb(face_align.norm_crop(im, kps[k]))
    names.append(disp); g_raw.append(r); g_adp.append(adapt(r))
    print(f"  [ok] enrolled {disp} from {fn}")
G_raw = np.array(g_raw, dtype=np.float32); G_adp = np.array(g_adp, dtype=np.float32)
print(f"[INFO] gallery: {', '.join(names)}\n")

def match(vecs, gal):
    sims = gal @ vecs; i = int(np.argmax(sims)); return names[i], float(sims[i])

# ---------- tracks with temporal aggregation ----------
class Trk:
    def __init__(self, tid, bbox, fn):
        self.id = tid; self.bbox = bbox; self.last = fn; self.n = 0
        self.sum_raw = np.zeros(G_raw.shape[1], np.float32)
        self.sum_adp = np.zeros(G_adp.shape[1], np.float32)
        self.pxs = []
    def add(self, r, a, px, fn):
        self.sum_raw += r; self.sum_adp += a; self.n += 1; self.pxs.append(px); self.last = fn
    def agg(self):
        ar = self.sum_adp / (np.linalg.norm(self.sum_adp) + 1e-9)
        rr = self.sum_raw / (np.linalg.norm(self.sum_raw) + 1e-9)
        return match(ar, G_adp), match(rr, G_raw)   # (adapter (name,sim)), (raw (name,sim))

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

tracks = {}; nid = 0
rows = []
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    print(f"[ERROR] cannot open {VIDEO}"); raise SystemExit
fnum = 0
print(f"[INFO] running {os.path.basename(VIDEO)} — press q to stop\n")
while True:
    ok, frame = cap.read()
    if not ok: break
    fnum += 1; disp = frame.copy()
    if fnum % FRAME_SKIP == 0:
        det_in = enhance_det(frame)
        faces = []
        for s in SCALES:
            scaled = det_in if s == 1.0 else cv2.resize(det_in, None, fx=s, fy=s)
            bb, kps = det_model.detect(scaled, max_num=0, metric='default')
            if bb is None: continue
            for k in range(len(bb)):
                box = bb[k, :4] / (s if s != 1.0 else 1.0)
                kp = kps[k] / (s if s != 1.0 else 1.0)
                if bb[k, 4] >= MIN_DET_SCORE:
                    faces.append((box, kp, float(bb[k, 4])))
        # simple NMS
        faces.sort(key=lambda f: f[2], reverse=True); kept = []
        for f in faces:
            if not any(iou(f[0], k[0]) > 0.5 for k in kept): kept.append(f)
        for box, kp, sc in kept:
            x1, y1, x2, y2 = [int(v) for v in box]
            px = min(x2 - x1, y2 - y1)
            if px < MIN_FACE_SIZE: continue
            crop = face_align.norm_crop(frame, kp)
            r = raw_emb(crop); a = adapt(r)
            # track match by IoU
            bi, bt = 0, None
            for tid, t in tracks.items():
                v = iou((x1, y1, x2, y2), t.bbox)
                if v > bi and v > IOU_THRESHOLD: bi, bt = v, tid
            if bt is None:
                bt = nid; tracks[nid] = Trk(nid, (x1, y1, x2, y2), fnum); nid += 1
            t = tracks[bt]; t.bbox = (x1, y1, x2, y2); t.add(r, a, px, fnum)
            (an, asim), (rn, rsim) = t.agg()
            rows.append({'frame': fnum, 'track': bt, 'px': px,
                         'adp_pred': an, 'adp_sim': round(asim, 3),
                         'raw_pred': rn, 'raw_sim': round(rsim, 3)})
    # draw persistent tracks EVERY frame (prevents box flicker)
    for t in tracks.values():
        if t.n < 2 or fnum - t.last > 10:
            continue
        (an, asim), (rn, rsim) = t.agg()
        # display RAW w600k (generalizes to unseen people); adapter overfits the 5 trained IDs
        label = rn if rsim >= SIM_THRESHOLD else "Unknown"
        col = (0, 255, 0) if rsim >= SIM_THRESHOLD else (0, 0, 255)
        x1, y1, x2, y2 = t.bbox
        cv2.rectangle(disp, (x1, y1), (x2, y2), col, 2)
        cv2.putText(disp, f"{label} {rsim:.2f} {min(x2 - x1, y2 - y1)}px",
                    (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    d = disp
    if d.shape[1] > DISPLAY_MAX_W:
        s = DISPLAY_MAX_W / d.shape[1]; d = cv2.resize(d, None, fx=s, fy=s)
    cv2.imshow("08 benchmark", d)
    if (cv2.waitKey(1) & 0xFF) in [ord('q'), 27]: break
cap.release(); cv2.destroyAllWindows()

# ---------- ground truth via distance ordering ----------
# Drone approaches; ordering is constant: Tejeswar closest -> Krishna -> Abhinav.
# On screen, closer = larger face, so mean-px rank gives each track's true identity.
GT_ORDER = ["Tejeswar", "Krishna", "Abhinav"]     # largest px -> smallest px
STAGE_METERS = [
    {"Tejeswar": 6, "Krishna": 7, "Abhinav": 8},  # stage 0 (drone 0m)
    {"Tejeswar": 5, "Krishna": 6, "Abhinav": 7},  # stage 1 (+1m)
    {"Tejeswar": 4, "Krishna": 5, "Abhinav": 6},  # stage 2 (+2m)
    {"Tejeswar": 3, "Krishna": 4, "Abhinav": 5},  # stage 3 (+3m)
]
total = max(fnum, 1)
def stage_of(frame):
    return min(3, int((frame - 1) / total * 4))

main = sorted([t for t in tracks.values() if t.n >= 3], key=lambda x: -x.n)[:len(GT_ORDER)]
main = sorted(main, key=lambda t: -float(np.mean(t.pxs)))   # largest mean px = closest = Tejeswar
gt = {t.id: GT_ORDER[i] for i, t in enumerate(main)}

print("\n" + "=" * 82)
print(f"PER-TRACK RESULT  (threshold {SIM_THRESHOLD})   frames processed: {fnum}")
print("=" * 82)
print(f"  {'trk':>3} {'GT':>9} {'frames':>6} {'px_range':>11} | {'ADAPTER':>9} {'sim':>5} {'ok':>3} | {'RAW':>9} {'sim':>5} {'ok':>3}")
acc_ok = acc_tot = 0
for t in sorted(main, key=lambda x: -float(np.mean(x.pxs))):
    (an, asim), (rn, rsim) = t.agg()
    truth = gt[t.id]
    ad = an if asim >= SIM_THRESHOLD else "Unknown"
    rd = rn if rsim >= SIM_THRESHOLD else "Unknown"
    a_ok = "Y" if ad == truth else "n"; r_ok = "Y" if rd == truth else "n"
    acc_ok += rd == truth; acc_tot += 1
    print(f"  {t.id:>3} {truth:>9} {t.n:>6} {min(t.pxs):>4}-{max(t.pxs):<4}px | "
          f"{ad:>9} {asim:>5.2f} {a_ok:>3} | {rd:>9} {rsim:>5.2f} {r_ok:>3}")
print(f"\n  Per-track identification (RAW): {acc_ok}/{acc_tot} correct")

print(f"\n-- Per-frame accuracy vs DISTANCE (adapter vs raw, threshold {SIM_THRESHOLD}) --")
print(f"  {'meters':>7} {'n':>6} {'adapter_acc':>12} {'raw_acc':>9}  (mean sim adp/raw)")
for m in [3, 4, 5, 6, 7, 8]:
    b = [r for r in rows if r['track'] in gt and STAGE_METERS[stage_of(r['frame'])][gt[r['track']]] == m]
    if not b: continue
    a_ok = sum(1 for r in b if r['adp_pred'] == gt[r['track']] and r['adp_sim'] >= SIM_THRESHOLD)
    r_ok = sum(1 for r in b if r['raw_pred'] == gt[r['track']] and r['raw_sim'] >= SIM_THRESHOLD)
    print(f"  {m:>6}m {len(b):>6} {a_ok/len(b):>11.1%} {r_ok/len(b):>9.1%}"
          f"   ({np.mean([r['adp_sim'] for r in b]):.2f}/{np.mean([r['raw_sim'] for r in b]):.2f})")

for r in rows:
    r['gt'] = gt.get(r['track'], '')
    r['meters'] = STAGE_METERS[stage_of(r['frame'])].get(r['gt'], '') if r['gt'] else ''
os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['frame', 'track', 'gt', 'meters', 'px',
                                       'adp_pred', 'adp_sim', 'raw_pred', 'raw_sim'])
    w.writeheader(); w.writerows(rows)
print(f"\n[SAVED] per-frame CSV -> {CSV_OUT}")
print("[DONE]")
