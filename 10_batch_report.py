"""
10_batch_report.py — run the consolidated fr2 pipeline over ALL videos, headless,
and write a DETAILED CSV of every recognition parameter.
Pipeline: buffalo_l detect (dark-enhance + multiscale, det_size 1280) -> norm_crop ->
bilateral denoise -> w600k embed -> IoU track -> temporal aggregation -> gallery match.
Outputs:
  reports/batch_detections.csv  (one row per detected face per processed frame)
  reports/batch_tracks.csv      (one row per track per video = final identity)
  reports/batch_params.csv      (the config used)
"""
import os
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, csv, glob
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ---------- config ----------
VIDEO_DIR   = r"C:\Users\hticd\OneDrive\Desktop\DJI data\DJI Avata & Mini 4K"
KNOWN_DIR   = r"C:\dev\face-recognition-pipeline\Known_faces"
OUT_DIR     = r"C:\dev\train_val\reports"
FRAME_SKIP  = 5
DET_SIZE    = 1280
SCALES      = [0.5, 1.0, 2.0]
MIN_FACE    = 12
MIN_DET     = 0.15
IOU_THRESH  = 0.2
THRESHOLD   = 0.40
USE_DENOISE = True
EXT_VID = ('.mp4', '.mov', '.avi', '.mkv')
EXT_IMG = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

os.makedirs(OUT_DIR, exist_ok=True)
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
det = app.models['detection']; rec = app.models['recognition']
print(f"[INFO] providers: {rec.session.get_providers()}")

def embed(face112):
    if USE_DENOISE:
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

# ---------- enroll: one clean Known_faces photo per person (beat multi-image averaging on the videos) ----------
names, gal = [], []
for f in sorted(os.listdir(KNOWN_DIR)):
    if not f.lower().endswith(EXT_IMG): continue
    im = cv2.imread(os.path.join(KNOWN_DIR, f))
    if im is None: continue
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0: print(f"  [skip] no face in {f}"); continue
    k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
    names.append(os.path.splitext(f)[0]); gal.append(embed(face_align.norm_crop(im, kps[k])))
    print(f"  [ok] {os.path.splitext(f)[0]}: single photo")
G = np.array(gal, np.float32)
print(f"[INFO] enrolled: {', '.join(names)}\n")

def match(v):
    s = G @ v; i = int(np.argmax(s)); o = np.argsort(s)[::-1]
    j = int(o[1]) if len(o) > 1 else i
    return names[i], float(s[i]), names[j], float(s[j])

# ---------- process each video ----------
videos = [f for f in sorted(os.listdir(VIDEO_DIR)) if f.lower().endswith(EXT_VID)]
det_rows, trk_rows = [], []
for vid in videos:
    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, vid))
    if not cap.isOpened(): print(f"[skip] cannot open {vid}"); continue
    tracks = {}; nid = 0; fnum = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        fnum += 1
        if fnum % FRAME_SKIP: continue
        det_in = enhance_det(frame)
        faces = []
        for s in SCALES:
            sc = det_in if s == 1.0 else cv2.resize(det_in, None, fx=s, fy=s)
            bb, kps = det.detect(sc, max_num=0, metric='default')
            if bb is None: continue
            for k in range(len(bb)):
                if bb[k, 4] < MIN_DET: continue
                faces.append((bb[k, :4] / s, kps[k] / s, float(bb[k, 4])))
        faces.sort(key=lambda x: x[2], reverse=True); kept = []
        for f in faces:
            if not any(iou(f[0], k[0]) > 0.5 for k in kept): kept.append(f)
        for box, kp, sc in kept:
            x1, y1, x2, y2 = [int(v) for v in box]
            px = min(x2-x1, y2-y1)
            if px < MIN_FACE: continue
            crop = face_align.norm_crop(frame, kp)
            e = embed(crop)
            fn_pred, fn_sim, _, _ = match(e)
            bi, bt = 0, None
            for tid, t in tracks.items():
                v = iou((x1, y1, x2, y2), t['bbox'])
                if v > bi and v > IOU_THRESH: bi, bt = v, tid
            if bt is None:
                bt = nid; tracks[nid] = {'bbox': (x1, y1, x2, y2), 'sum': e.copy(), 'n': 1,
                                         'pxs': [px], 'last': fnum, 'best_crop': crop, 'best_px': px}; nid += 1
            else:
                t = tracks[bt]; t['bbox'] = (x1, y1, x2, y2); t['sum'] += e; t['n'] += 1
                t['pxs'].append(px); t['last'] = fnum
                if px > t['best_px']: t['best_px'] = px; t['best_crop'] = crop
            t = tracks[bt]
            agg = t['sum'] / (np.linalg.norm(t['sum']) + 1e-9)
            ap, asim, a2, a2s = match(agg)
            det_rows.append({'video': vid, 'frame': fnum, 'track': bt, 'px': px,
                             'det_score': round(sc, 3), 'frame_pred': fn_pred, 'frame_sim': round(fn_sim, 3),
                             'agg_pred': ap, 'agg_sim': round(asim, 3), 'n_agg': t['n'],
                             'runner_up': a2, 'runner_sim': round(a2s, 3),
                             'decision': ap if asim >= THRESHOLD else 'Unknown'})
    for tid, t in tracks.items():
        if t['n'] < 2: continue
        agg = t['sum'] / (np.linalg.norm(t['sum']) + 1e-9)
        ap, asim, a2, a2s = match(agg)
        # save the crop labeled with what fr2 recognized it as (accepted -> name folder, else Unknown)
        label = ap if asim >= THRESHOLD else "Unknown"
        _d = os.path.join(OUT_DIR, "recognized", label); os.makedirs(_d, exist_ok=True)
        _stem = os.path.splitext(vid)[0][:22]
        cv2.imwrite(os.path.join(_d, f"{_stem}_trk{tid}_as-{ap}_sim{asim:.2f}.jpg"), t['best_crop'])
        trk_rows.append({'video': vid, 'track': tid, 'frames': t['n'],
                         'px_min': min(t['pxs']), 'px_max': max(t['pxs']),
                         'final_pred': ap, 'final_sim': round(asim, 3),
                         'runner_up': a2, 'runner_sim': round(a2s, 3),
                         'decision': ap if asim >= THRESHOLD else 'Unknown'})
    cap.release()
    print(f"[done] {vid:45s} tracks={len([t for t in tracks.values() if t['n']>=2])}")

# ---------- write CSVs (robust: never crash, fall back if the file is locked/open in Excel) ----------
import datetime
def _write_csv(fname, header, rows_writer):
    path = os.path.join(OUT_DIR, fname)
    try:
        with open(path, 'w', newline='') as fh:
            rows_writer(fh)
        print(f"[SAVED] {path}")
    except PermissionError:
        alt = path.replace('.csv', datetime.datetime.now().strftime('_%H%M%S.csv'))
        with open(alt, 'w', newline='') as fh:
            rows_writer(fh)
        print(f"[LOCKED] {fname} is open (close Excel!) -> wrote {os.path.basename(alt)} instead.")

if not det_rows:
    print("[WARN] no detections — nothing to write (did the run finish / find faces?)")
else:
    _write_csv("batch_detections.csv", None,
               lambda fh: (lambda w: (w.writeheader(), w.writerows(det_rows)))(csv.DictWriter(fh, fieldnames=list(det_rows[0].keys()))))
    _write_csv("batch_tracks.csv", None,
               lambda fh: (lambda w: (w.writeheader(), w.writerows(trk_rows)))(csv.DictWriter(fh, fieldnames=list(trk_rows[0].keys()))))
    def _params(fh):
        w = csv.writer(fh); w.writerow(['param', 'value'])
        for k, v in [('detector', 'buffalo_l'), ('det_size', DET_SIZE), ('scales', SCALES),
                     ('frame_skip', FRAME_SKIP), ('min_det_score', MIN_DET), ('iou_thresh', IOU_THRESH),
                     ('sim_threshold', THRESHOLD), ('denoise', USE_DENOISE), ('enrolled', ';'.join(names))]:
            w.writerow([k, v])
    _write_csv("batch_params.csv", None, _params)
print(f"\n[DONE] {len(det_rows)} detections, {len(trk_rows)} tracks -> {OUT_DIR}")
