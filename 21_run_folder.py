"""
21_run_folder.py — run recognition over drone clips WITH visual verification.

    python 21_run_folder.py                          # whole DJI_001 folder
    python 21_run_folder.py "C:\\path\\clip.MP4"      # one clip
    python 21_run_folder.py --show                   # + live preview window (q = skip clip, ESC = quit)

Outputs -> reports/dji_run/<folder>/
    <clip>_annotated.mp4              full video with boxes + name + sim + px
    recognized/<Name>/*.jpg           every face called that name (filename carries sim + px)
    recognized/Unknown/*.jpg          faces that were detected but not confidently named
    sheets/_SHEET_<Name>.jpg          CONTACT SHEET: enrolled REFERENCE first, then every face
                                      claimed to be that person, sorted by confidence
    tracks.csv                        one row per track

The contact sheets are the visual audit: if a face in <Name>'s sheet is not <Name>, that is a
false accept; if someone you know appears in Unknown's sheet, that is a miss.

Notes:
  * .LRF files (DJI low-res proxies) are skipped automatically.
  * TILED DETECTION: SCRFD resizes any frame down to det_size, so a 40px face in 4K becomes ~13px
    and disappears. We also detect on overlapping tiles at native resolution and map boxes back.
  * Gallery accepts Known_faces/<Name>.jpg OR Known_faces/<Name>/*.jpg (multi-reference, averaged).
"""
import os, sys
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, csv
from collections import defaultdict
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ---------------- config ----------------
DEFAULT_DIR = r"C:\Users\hticd\OneDrive\Desktop\DJI data\DJI_001"
KNOWN_DIR   = r"C:\dev\face-recognition-pipeline\Known_faces"
COHORT_NPZ  = r"C:\dev\train_val\models\cohort.npz"   # AS-Norm pack (26_build_cohort.py)
OUT_ROOT    = r"C:\dev\train_val\reports\dji_run"
FRAME_SKIP  = 10
DET_SIZE    = 1280
MIN_DET     = 0.30
IOU_THRESH  = 0.25
THRESHOLD   = 0.42        # 0.36 recovers 3 correct IDs on DJI_001 (0.39/0.39/0.38) with 0 false
                          # accepts THERE — but the impostor set shows stranger-rejection falls
                          # 99.4% -> ~97.5% at 0.36. Multi-reference enrollment is the better fix:
                          # it raises genuine sims ~+0.20 so you keep 0.40 AND catch these faces.
MIN_FACE_PX = 20          # below this the detector produces junk (the 12px/14px non-face tiles);
                          # smallest CORRECT face in DJI_001 was 28px, so this costs nothing
MIN_FRAMES  = 3
USE_DENOISE = True
TILE_IF_WIDER_THAN = 2000
TILE_GRID, TILE_OVERLAP = (2, 2), 0.20
ANNOT_WIDTH = 1600
CONTEXT_PAD = 0.6          # save crops with this much context around the face box
K_CAMERA    = None         # set to show metres: d = K / face_px

VID_EXT  = ('.mp4', '.mov', '.avi', '.mkv')
SKIP_EXT = ('.lrf', '.srt', '.txt')
IMG_EXT  = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

SHOW = '--show' in sys.argv
posargs = [a for a in sys.argv[1:] if not a.startswith('--')]

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
det, rec = app.models['detection'], app.models['recognition']

def embed(face112):
    if USE_DENOISE: face112 = cv2.bilateralFilter(face112, 5, 50, 50)
    f = rec.get_feat(face112).flatten().astype(np.float32)
    return f / (np.linalg.norm(f) + 1e-9)

_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
def enhance_det(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    out = cv2.cvtColor(cv2.merge([_clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    m = out.mean()
    if m < 110:
        g = 1.6 if m < 70 else 1.3
        lut = np.array([((i/255.0)**(1.0/g))*255 for i in range(256)]).astype('uint8')
        out = cv2.LUT(out, lut)
    return out

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1]); x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 < x1 or y2 < y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-6)

def context_crop(frame, box, pad=CONTEXT_PAD):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx1 = max(0, int(x1 - w*pad)); cy1 = max(0, int(y1 - h*pad))
    cx2 = min(W, int(x2 + w*pad)); cy2 = min(H, int(y2 + h*pad))
    c = frame[cy1:cy2, cx1:cx2]
    return c if c.size else None

# ---------------- gallery ----------------
names, embs, ref_thumb = [], [], {}
print("[INFO] enrolling...")
for entry in sorted(os.listdir(KNOWN_DIR)):
    full = os.path.join(KNOWN_DIR, entry)
    if os.path.isdir(full) and not entry.startswith('_'):
        person = entry
        files = [os.path.join(full, f) for f in sorted(os.listdir(full)) if f.lower().endswith(IMG_EXT)]
    elif entry.lower().endswith(IMG_EXT):
        person = os.path.splitext(entry)[0]; files = [full]
    else:
        continue
    vecs = []
    for p in files:
        im = cv2.imread(p)
        if im is None: continue
        bb, kps = det.detect(im, max_num=0, metric='default')
        if bb is None or len(bb) == 0: continue
        k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
        crop = face_align.norm_crop(im, kps[k])
        vecs.append(embed(crop))
        if person not in ref_thumb: ref_thumb[person] = crop
    if not vecs:
        print(f"  [skip] {person}: no usable reference"); continue
    m = np.mean(vecs, axis=0); m /= (np.linalg.norm(m) + 1e-9)
    names.append(person); embs.append(m.astype(np.float32))
    print(f"  [ok] {person:14s} {len(vecs)} reference(s)")
G = np.array(embs, np.float32)
print(f"[INFO] enrolled {len(names)}: {', '.join(names)}\n")

# ---------------- AS-Norm: score relative to THIS probe's stranger floor ----------------
# A degraded face scores low against everything, so a fixed cosine bar rejects correct matches.
# Normalising by the probe's own top-K cohort scores removes that depression.
COH, TOPK, Z_THRESH = None, 200, 0.0
if os.path.exists(COHORT_NPZ):
    _c = np.load(COHORT_NPZ, allow_pickle=True)
    COH = _c['cohort'].astype(np.float32)
    TOPK = int(_c['topk']); Z_THRESH = float(_c['z_thresh'])
    print(f"[INFO] AS-Norm ON  — cohort {COH.shape[0]} faces, K={TOPK}, "
          f"z_thresh={Z_THRESH:.3f}, domain={str(_c['domain'])}")
else:
    print(f"[INFO] AS-Norm OFF — no cohort at {COHORT_NPZ}; using raw cosine >= {THRESHOLD}")

def score(v):
    """-> name, raw_sim, runner_up, runner_sim, z (None if no cohort), accepted"""
    s = G @ v
    o = np.argsort(s)[::-1]
    i = int(o[0]); j = int(o[1]) if len(o) > 1 else i
    z = None
    if COH is not None:
        c = COH @ v
        t = np.partition(c, -TOPK)[-TOPK:]
        z = float((s[i] - t.mean()) / (t.std() + 1e-6))
        acc = z >= Z_THRESH
    else:
        acc = s[i] >= THRESHOLD
    return names[i], float(s[i]), names[j], float(s[j]), z, acc

def detect_all(img, use_tiles):
    cand = []
    bb, kps = det.detect(img, max_num=0, metric='default')
    if bb is not None:
        for i in range(len(bb)):
            if bb[i, 4] >= MIN_DET: cand.append((bb[i, :4].copy(), kps[i].copy(), float(bb[i, 4])))
    if use_tiles:
        H, W = img.shape[:2]; R, C = TILE_GRID
        th, tw = H // R, W // C
        oy, ox = int(th*TILE_OVERLAP), int(tw*TILE_OVERLAP)
        for r in range(R):
            for c in range(C):
                y1 = max(0, r*th-oy); y2 = min(H, (r+1)*th+oy)
                x1 = max(0, c*tw-ox); x2 = min(W, (c+1)*tw+ox)
                tb, tk = det.detect(img[y1:y2, x1:x2], max_num=0, metric='default')
                if tb is None: continue
                for i in range(len(tb)):
                    if tb[i, 4] < MIN_DET: continue
                    b = tb[i, :4].copy(); b[0] += x1; b[1] += y1; b[2] += x1; b[3] += y1
                    k = tk[i].copy(); k[:, 0] += x1; k[:, 1] += y1
                    cand.append((b, k, float(tb[i, 4])))
    cand.sort(key=lambda t: -t[2])
    kept = []
    for c in cand:
        if not any(iou(c[0], k[0]) > 0.4 for k in kept): kept.append(c)
    return kept

# ---------------- clips ----------------
target = posargs[0] if posargs else DEFAULT_DIR
if os.path.isfile(target):
    clips = [target]; src_dir = os.path.dirname(target)
else:
    src_dir = target
    clips = [os.path.join(src_dir, f) for f in sorted(os.listdir(src_dir))
             if f.lower().endswith(VID_EXT) and not f.lower().endswith(SKIP_EXT)]
out_dir = os.path.join(OUT_ROOT, os.path.basename(src_dir.rstrip('\\/')))
crop_dir = os.path.join(out_dir, "recognized"); sheet_dir = os.path.join(out_dir, "sheets")
os.makedirs(crop_dir, exist_ok=True); os.makedirs(sheet_dir, exist_ok=True)
print(f"[INFO] {len(clips)} clip(s); output -> {out_dir}\n")

all_rows = []
gallery_of = defaultdict(list)     # name -> [(sim, crop112, context, tag)]
abort = False
for path in clips:
    if abort: break
    clip = os.path.basename(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): print(f"[skip] cannot open {clip}"); continue
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    use_tiles = W > TILE_IF_WIDER_THAN
    print(f"=== {clip}  {W}x{H} @ {fps:.1f}fps  {total} frames ({total/fps:.0f}s)  "
          f"tiling={'ON' if use_tiles else 'off'}")
    scale = ANNOT_WIDTH / W
    writer = cv2.VideoWriter(os.path.join(out_dir, os.path.splitext(clip)[0]+"_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*'mp4v'), max(1.0, fps/FRAME_SKIP),
                             (int(W*scale), int(H*scale)))
    tracks = {}; nid = 0; fnum = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        fnum += 1
        if fnum % FRAME_SKIP: continue
        faces = detect_all(enhance_det(frame), use_tiles)
        vis = cv2.resize(frame, (int(W*scale), int(H*scale)))
        for box, kp, sc in faces:
            x1, y1, x2, y2 = [int(v) for v in box]
            px = min(x2-x1, y2-y1)
            if px < MIN_FACE_PX: continue
            crop112 = face_align.norm_crop(frame, kp)
            e = embed(crop112)
            best, bi = 0.0, None
            for tid, t in tracks.items():
                v = iou((x1, y1, x2, y2), t['bbox'])
                if v > best and v > IOU_THRESH: best, bi = v, tid
            if bi is None:
                bi = nid
                tracks[nid] = {'bbox': (x1, y1, x2, y2), 'sum': e.copy(), 'n': 1, 'pxs': [px],
                               'best_px': px, 'crop': crop112,
                               'ctx': context_crop(frame, (x1, y1, x2, y2)), 'frame': fnum}
                nid += 1
            else:
                t = tracks[bi]; t['bbox'] = (x1, y1, x2, y2); t['sum'] += e
                t['n'] += 1; t['pxs'].append(px)
                if px > t['best_px']:
                    t['best_px'] = px; t['crop'] = crop112
                    t['ctx'] = context_crop(frame, (x1, y1, x2, y2)); t['frame'] = fnum
            t = tracks[bi]
            agg = t['sum'] / (np.linalg.norm(t['sum']) + 1e-9)
            nm, sim, _, _, z, acc = score(agg)
            label = nm if acc else "Unknown"
            col = (0, 200, 0) if acc else (40, 40, 220)
            zt = f" z{z:.1f}" if z is not None else ""
            txt = f"{label} {sim:.2f}{zt} {px}px" + (f" ~{K_CAMERA/max(px,1):.1f}m" if K_CAMERA else "")
            p1 = (int(x1*scale), int(y1*scale)); p2 = (int(x2*scale), int(y2*scale))
            cv2.rectangle(vis, p1, p2, col, 2)
            cv2.rectangle(vis, (p1[0], max(0, p1[1]-20)), (p1[0]+len(txt)*8, p1[1]), col, -1)
            cv2.putText(vis, txt, (p1[0]+2, max(13, p1[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1)
        writer.write(vis)
        if SHOW:
            cv2.imshow("recognition (q=next clip, ESC=quit)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k == 27: abort = True; break
    cap.release(); writer.release()

    seen = defaultdict(list)
    for tid, t in tracks.items():
        if t['n'] < MIN_FRAMES: continue
        agg = t['sum'] / (np.linalg.norm(t['sum']) + 1e-9)
        nm, sim, n2, s2, z, acc = score(agg)
        dec = nm if acc else "Unknown"
        all_rows.append({'clip': clip, 'track': tid, 'frames': t['n'],
                         'px_min': min(t['pxs']), 'px_max': max(t['pxs']), 'pred': nm,
                         'sim': round(sim, 3), 'z': ('' if z is None else round(z, 2)),
                         'runner_up': n2, 'runner_sim': round(s2, 3), 'decision': dec})
        d = os.path.join(crop_dir, dec); os.makedirs(d, exist_ok=True)
        zs = '' if z is None else f"_z{z:.1f}"
        stem = f"{os.path.splitext(clip)[0][:20]}_trk{tid}_as-{nm}_sim{sim:.2f}{zs}_{t['best_px']}px"
        img = t['ctx'] if t['ctx'] is not None and t['ctx'].size else t['crop']
        cv2.imwrite(os.path.join(d, stem + ".jpg"), img)
        lbl = f"{sim:.2f}" + ('' if z is None else f" z{z:.1f}") + f" {t['best_px']}px"
        gallery_of[dec].append((sim, t['crop'], lbl))
        seen[dec].append(sim)
    rep = ", ".join(f"{k}x{len(v)}(best {max(v):.2f})" for k, v in sorted(seen.items()) if k != "Unknown")
    print(f"    tracks={sum(1 for t in tracks.values() if t['n']>=MIN_FRAMES)}  "
          f"recognised: {rep or 'none'}  unknown={len(seen.get('Unknown', []))}\n")
if SHOW: cv2.destroyAllWindows()

# ---------------- contact sheets (the visual audit) ----------------
def build_sheet(person, items, ref, path):
    CELL, PAD, COLS, LBL = 128, 8, 8, 24
    tiles = []
    if ref is not None: tiles.append((cv2.resize(ref, (CELL, CELL)), "REFERENCE", True))
    for sim, crop, lab in items:
        tiles.append((cv2.resize(crop, (CELL, CELL)), lab, False))
    rows = max(1, (len(tiles) + COLS - 1) // COLS)
    sheet = np.full((rows*(CELL+LBL) + PAD*2 + 30, COLS*(CELL+PAD) + PAD, 3), 28, np.uint8)
    cv2.putText(sheet, f"{person}  —  {len(items)} face(s) called this name",
                (PAD, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
    for i, (im, lab, is_ref) in enumerate(tiles):
        r, c = i // COLS, i % COLS
        y = 30 + PAD + r*(CELL+LBL); x = PAD + c*(CELL+PAD)
        sheet[y:y+CELL, x:x+CELL] = im
        colr = (0, 215, 255) if is_ref else (0, 200, 0)
        cv2.rectangle(sheet, (x, y), (x+CELL, y+CELL), colr, 2)
        cv2.putText(sheet, lab, (x, y+CELL+16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1)
    cv2.imwrite(path, sheet)

for person, items in gallery_of.items():
    items.sort(key=lambda t: -t[0])
    build_sheet(person, items, ref_thumb.get(person), os.path.join(sheet_dir, f"_SHEET_{person}.jpg"))

# ---------------- summary ----------------
if all_rows:
    with open(os.path.join(out_dir, "tracks.csv"), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    named = [r for r in all_rows if r['decision'] != 'Unknown']
    print("=" * 66)
    print(f"TOTAL tracks={len(all_rows)}  named={len(named)}  unknown={len(all_rows)-len(named)}")
    per = defaultdict(int)
    for r in named: per[r['decision']] += 1
    for k in sorted(per, key=lambda x: -per[x]): print(f"   {k:14s} {per[k]} track(s)")
    pxn = [r['px_max'] for r in named]; pxu = [r['px_max'] for r in all_rows if r['decision'] == 'Unknown']
    if pxn: print(f"   face px_max — named median {int(np.median(pxn))} | "
                  f"unknown median {int(np.median(pxu)) if pxu else 0}")
    print(f"\n[SAVED] crops   -> {crop_dir}\\<Name>\\")
    print(f"[SAVED] SHEETS  -> {sheet_dir}   <-- OPEN THESE to verify identities visually")
    print(f"[SAVED] video   -> {out_dir}\\*_annotated.mp4")
else:
    print("[WARN] no tracks — raise FRAME_SKIP tolerance or check face sizes.")
