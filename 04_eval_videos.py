"""
04_eval_videos.py — QUANTITATIVE EVALUATION ON DRONE VIDEOS
Enrolls a multi-image gallery from video_dataset/<Person>/train, then runs
recognition on held-out videos and reports:
  - genuine vs impostor separation -> precision/recall/F1 sweep + PR-AUC
  - accuracy vs distance (face pixel-size buckets)
  - per-video summary
Everything on GPU (insightface w600k). Leakage-guarded: crops from a genuine
test video are excluded from that person's gallery.
"""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")

import cv2
import numpy as np
import csv
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ==================== CONFIG ====================
ENROLL_DIR  = r"C:\dev\face-recognition-pipeline\video_dataset"
VIDEO_DIR   = r"C:\Users\hticd\OneDrive\Desktop\DJI data\DJI Avata & Mini 4K"
REPORT_CSV  = r"C:\dev\train_val\reports\video_eval.csv"
MAX_ENROLL_PER_PERSON = 200
FRAME_STRIDE = 10               # sample every Nth frame of each test video
MIN_FACE_PX  = 20
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# People to EXCLUDE from the gallery (so you can test them as impostors / strangers)
ENROLL_EXCLUDE_PEOPLE = ["Dakshina"]     # -> Da.MOV becomes a clean impostor test below

# Held-out test videos: filename -> true label, or "IMPOSTOR" for a non-enrolled person.
# Use SINGLE-PERSON videos (largest face per frame = the subject).
TEST_VIDEOS = {
    "Nandana5.MP4":    "Nandana",
    "Nandana6.mp4":    "Nandana",
    "Siddhartha5.MP4": "Siddhartha",
    "Da.MOV":          "IMPOSTOR",       # Dakshina excluded from gallery -> should be rejected
    "Allen4.mp4":      "Allen",
}

THRESHOLDS = [round(x, 2) for x in np.arange(0.20, 0.85, 0.02)]
REPORT_THRESHOLD = 0.40          # threshold used for the per-distance table
PX_BUCKETS = [(0, 30), (30, 60), (60, 120), (120, 10 ** 9)]

# ==================== MODEL (GPU) ====================
print("[INFO] Loading detector + recognizer ...")
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1280, 1280))
det_model = app.models['detection']
rec_model = app.models['recognition']
print("[INFO] Ready\n")

def get_embedding(face112):
    emb = rec_model.get_feat(face112).flatten().astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)

# ==================== ENROLL (multi-image, leakage-guarded) ====================
test_stems = {os.path.splitext(v)[0] for v, lbl in TEST_VIDEOS.items() if lbl != "IMPOSTOR"}

print("[INFO] Enrolling gallery ...")
gallery_names, _gal = [], []
for person in sorted(os.listdir(ENROLL_DIR)):
    if person.startswith("_") or person in ENROLL_EXCLUDE_PEOPLE:
        continue
    pdir = os.path.join(ENROLL_DIR, person, "train")
    if not os.path.isdir(pdir):
        continue
    files = [f for f in sorted(os.listdir(pdir)) if f.lower().endswith(VALID_EXT)]
    # drop crops that came from a genuine test video (avoid leakage)
    files = [f for f in files if not any(f.startswith(s + "_") for s in test_stems)]
    if len(files) > MAX_ENROLL_PER_PERSON:
        step = len(files) / MAX_ENROLL_PER_PERSON
        files = [files[int(i * step)] for i in range(MAX_ENROLL_PER_PERSON)]
    embs = []
    for fn in files:
        img = cv2.imread(os.path.join(pdir, fn))
        if img is None:
            continue
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LANCZOS4)
        embs.append(get_embedding(img))
    if not embs:
        continue
    mean = np.mean(embs, axis=0); mean /= (np.linalg.norm(mean) + 1e-9)
    gallery_names.append(person); _gal.append(mean.astype(np.float32))
    print(f"  [ok] {person:12s}: {len(embs)} crops")
gallery_embs = np.array(_gal, dtype=np.float32)
if len(gallery_names) == 0:
    print("[ERROR] Empty gallery."); exit()
print(f"[INFO] Gallery: {', '.join(gallery_names)}\n")

def match(face112):
    emb = get_embedding(face112)
    sims = gallery_embs @ emb
    i = int(np.argmax(sims))
    return gallery_names[i], float(sims[i])

# ==================== RUN TEST VIDEOS ====================
records = []   # dict(video, true, pred, sim, px, impostor)
for vid, label in TEST_VIDEOS.items():
    path = os.path.join(VIDEO_DIR, vid)
    if not os.path.exists(path):
        print(f"[skip] missing: {vid}"); continue
    cap = cv2.VideoCapture(path)
    fi = 0; n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % FRAME_STRIDE == 0:
            bb, kps = det_model.detect(frame, max_num=0, metric='default')
            if bb is not None and len(bb) > 0:
                k = int(np.argmax((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))  # largest face
                x1, y1, x2, y2 = bb[k, :4]
                px = int(min(x2 - x1, y2 - y1))
                if px >= MIN_FACE_PX:
                    face = face_align.norm_crop(frame, kps[k])
                    pred, sim = match(face)
                    records.append({'video': vid, 'true': label, 'pred': pred,
                                    'sim': sim, 'px': px, 'impostor': label == "IMPOSTOR"})
                    n += 1
        fi += 1
    cap.release()
    print(f"[done] {vid:40s} label={label:12s} faces={n}")
print(f"\n[INFO] {len(records)} total detections\n")

# ==================== METRICS ====================
def metrics_at(t):
    TP = FP = FN = TN = 0
    for r in records:
        if r['impostor']:
            if r['sim'] >= t: FP += 1
            else: TN += 1
        else:
            if r['sim'] >= t and r['pred'] == r['true']: TP += 1
            elif r['sim'] >= t: FP += 1
            else: FN += 1
    p = TP / (TP + FP + 1e-9); rc = TP / (TP + FN + 1e-9)
    f1 = 2 * p * rc / (p + rc + 1e-9)
    return {'t': t, 'P': p, 'R': rc, 'F1': f1, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

rows = [metrics_at(t) for t in THRESHOLDS]
# PR-AUC
pairs = sorted([(r['R'], r['P']) for r in rows])
pr_auc = float(np.trapezoid([p for _, p in pairs], [r for r, _ in pairs]))
best_f1 = max(rows, key=lambda r: r['F1'])
best_p = max([r for r in rows if r['R'] >= 0.3] or rows, key=lambda r: r['P'])

print("=" * 74)
print("VIDEO EVALUATION REPORT")
print("=" * 74)
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Best-F1  @ t={best_f1['t']:.2f}: P={best_f1['P']:.3f} R={best_f1['R']:.3f} F1={best_f1['F1']:.3f}")
print(f"Best-Prec@ t={best_p['t']:.2f}: P={best_p['P']:.3f} R={best_p['R']:.3f} F1={best_p['F1']:.3f}")

print("\n-- Threshold sweep --")
print(f"  {'t':>5} {'Prec':>6} {'Rec':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
for r in rows[::2]:
    print(f"  {r['t']:>5.2f} {r['P']:>6.3f} {r['R']:>6.3f} {r['F1']:>6.3f} "
          f"{r['TP']:>5} {r['FP']:>5} {r['FN']:>5} {r['TN']:>5}")

print(f"\n-- Accuracy vs distance (genuine only, threshold {REPORT_THRESHOLD}) --")
gen = [r for r in records if not r['impostor']]
for lo, hi in PX_BUCKETS:
    b = [r for r in gen if lo <= r['px'] < hi]
    if not b:
        continue
    correct = sum(1 for r in b if r['sim'] >= REPORT_THRESHOLD and r['pred'] == r['true'])
    tag = f"{lo}-{'inf' if hi > 10**8 else hi}px"
    print(f"  {tag:>10}: {correct}/{len(b)} = {correct/len(b):.2%}  (mean sim {np.mean([r['sim'] for r in b]):.3f})")

print("\n-- Per-video --")
for vid in TEST_VIDEOS:
    b = [r for r in records if r['video'] == vid]
    if not b:
        continue
    if b[0]['impostor']:
        acc = sum(1 for r in b if r['sim'] < REPORT_THRESHOLD) / len(b)   # correctly rejected
        print(f"  {vid:40s} IMPOSTOR  rejected {acc:.2%}  (mean sim {np.mean([r['sim'] for r in b]):.3f})")
    else:
        acc = sum(1 for r in b if r['sim'] >= REPORT_THRESHOLD and r['pred'] == r['true']) / len(b)
        print(f"  {vid:40s} {b[0]['true']:12s} correct {acc:.2%}  (mean sim {np.mean([r['sim'] for r in b]):.3f})")

os.makedirs(os.path.dirname(REPORT_CSV), exist_ok=True)
with open(REPORT_CSV, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['video', 'true', 'pred', 'sim', 'px', 'impostor'])
    w.writeheader(); w.writerows(records)
print(f"\n[SAVED] per-detection CSV -> {REPORT_CSV}")
print("[DONE]")
