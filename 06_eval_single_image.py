"""
06_eval_single_image.py — SINGLE-IMAGE ENROLLMENT EVALUATION (watchlist reality)
Each person is enrolled from ONE reference image (like a single mugshot in a DB),
then tested on all hand-labeled crops in video_dataset/_unsorted/<video>/<Person>/.
This is the honest "identify a person from one photo" number.

ENROLL_MODE:
  "one_crop"    -> 1 crop per person from video_dataset/<Person>/train
                   (picks the crop nearest that person's centroid = a clean, typical shot).
                   Same 5 people as the multi-image eval -> direct comparison.
  "known_faces" -> 1 mugshot per person from Known_faces/<Name>.jpg (detect+align).
                   Truest "one DB photo"; includes Abhinav1 even if his train is empty.
"""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2, numpy as np, shutil, csv
from collections import defaultdict
from insightface.app import FaceAnalysis
from insightface.utils import face_align

ENROLL_MODE = "one_crop"          # "one_crop" | "known_faces"
DS          = r"C:\dev\face-recognition-pipeline\video_dataset"
UNS         = os.path.join(DS, "_unsorted")
KNOWN_DIR   = r"C:\dev\face-recognition-pipeline\Known_faces"
FAIL_DIR    = r"C:\dev\train_val\reports\failures_single"
CSV_OUT     = r"C:\dev\train_val\reports\single_eval.csv"
THRESHOLD   = 0.40
CENTROID_CAP = 200                # crops used only to PICK the representative one
EXT = ('.jpg', '.jpeg', '.png', '.bmp')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det_model = app.models['detection']
rec = app.models['recognition']

def emb(img):
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112))
    e = rec.get_feat(img).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

# ---------- SINGLE-IMAGE enrollment ----------
print(f"[INFO] enrolling ONE image per person (mode={ENROLL_MODE}) ...")
gallery_names, _gal = [], []

if ENROLL_MODE == "one_crop":
    for person in sorted(os.listdir(DS)):
        if person.startswith("_"):
            continue
        pdir = os.path.join(DS, person, "train")
        if not os.path.isdir(pdir):
            continue
        files = [f for f in sorted(os.listdir(pdir)) if f.lower().endswith(EXT)]
        if not files:
            continue
        sample = files
        if len(files) > CENTROID_CAP:
            step = len(files) / CENTROID_CAP
            sample = [files[int(i * step)] for i in range(CENTROID_CAP)]
        es, keep = [], []
        for f in sample:
            im = cv2.imread(os.path.join(pdir, f))
            if im is not None:
                es.append(emb(im)); keep.append(f)
        if not es:
            continue
        M = np.array(es)
        centroid = M.mean(axis=0); centroid /= (np.linalg.norm(centroid) + 1e-9)
        best = int(np.argmax(M @ centroid))        # crop closest to centroid = representative single shot
        gallery_names.append(person); _gal.append(es[best])
        print(f"  [ok] {person:12s} <- 1 crop ({keep[best]})")
else:  # known_faces
    for f in sorted(os.listdir(KNOWN_DIR)):
        if not f.lower().endswith(EXT):
            continue
        name = os.path.splitext(f)[0]
        im = cv2.imread(os.path.join(KNOWN_DIR, f))
        if im is None:
            continue
        faces = det_model.detect(im, max_num=0, metric='default')
        bb, kps = faces
        if bb is None or len(bb) == 0:
            print(f"  [skip] no face in {f}"); continue
        k = int(np.argmax((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))
        aligned = face_align.norm_crop(im, kps[k])
        gallery_names.append(name); _gal.append(emb(aligned))
        print(f"  [ok] {name:12s} <- {f}")

gallery_embs = np.array(_gal, dtype=np.float32)
if not gallery_names:
    print("[ERROR] empty gallery."); raise SystemExit
print(f"[INFO] gallery ({len(gallery_names)}): {', '.join(gallery_names)}\n")

# ---------- test on labeled crops ----------
records = []
for cond in sorted(os.listdir(UNS)):
    cpath = os.path.join(UNS, cond)
    if not os.path.isdir(cpath):
        continue
    for person in sorted(os.listdir(cpath)):
        if person.startswith("_"):
            continue
        ppath = os.path.join(cpath, person)
        if not os.path.isdir(ppath):
            continue
        for f in os.listdir(ppath):
            if not f.lower().endswith(EXT):
                continue
            im = cv2.imread(os.path.join(ppath, f))
            if im is None:
                continue
            e = emb(im)
            sims = gallery_embs @ e
            i = int(np.argmax(sims))
            records.append({'cond': cond, 'true': person, 'pred': gallery_names[i],
                            'sim': float(sims[i]), 'path': os.path.join(ppath, f),
                            'impostor': person not in gallery_names})
if not records:
    print("[INFO] no labeled crops found."); raise SystemExit
print(f"[INFO] {len(records)} test crops\n")

# ---------- metrics ----------
def sweep(t):
    TP = FP = FN = TN = 0
    for r in records:
        if r['impostor']:
            FP += r['sim'] >= t; TN += r['sim'] < t
        else:
            if r['sim'] >= t and r['pred'] == r['true']: TP += 1
            elif r['sim'] >= t: FP += 1
            else: FN += 1
    p = TP / (TP + FP + 1e-9); rc = TP / (TP + FN + 1e-9)
    return {'t': t, 'P': p, 'R': rc, 'F1': 2 * p * rc / (p + rc + 1e-9), 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}

rows = [sweep(round(x, 2)) for x in np.arange(0.20, 0.85, 0.02)]
pairs = sorted((r['R'], r['P']) for r in rows)
pr_auc = float(np.trapezoid([p for _, p in pairs], [r for r, _ in pairs]))
best = max(rows, key=lambda r: r['F1'])

print("=" * 70)
print(f"SINGLE-IMAGE EVALUATION (mode={ENROLL_MODE})")
print("=" * 70)
print(f"PR-AUC: {pr_auc:.4f}   Best-F1 @ t={best['t']:.2f}: "
      f"P={best['P']:.3f} R={best['R']:.3f} F1={best['F1']:.3f}")

print(f"\n-- Per-condition accuracy (genuine, t={THRESHOLD}) --")
bc = defaultdict(list)
for r in records:
    if not r['impostor']:
        bc[r['cond']].append(r)
for c in sorted(bc):
    b = bc[c]; ok = sum(1 for r in b if r['sim'] >= THRESHOLD and r['pred'] == r['true'])
    print(f"  {c[:38]:38s} {ok}/{len(b)} = {ok/len(b):6.1%}  (mean sim {np.mean([r['sim'] for r in b]):.3f})")

print(f"\n-- Per-person accuracy (genuine, t={THRESHOLD}) --")
bp = defaultdict(list)
for r in records:
    if not r['impostor']:
        bp[r['true']].append(r)
for p in sorted(bp):
    b = bp[p]; ok = sum(1 for r in b if r['sim'] >= THRESHOLD and r['pred'] == r['true'])
    print(f"  {p:12s} {ok}/{len(b)} = {ok/len(b):6.1%}")

imp = [r for r in records if r['impostor']]
if imp:
    rej = sum(1 for r in imp if r['sim'] < THRESHOLD)
    print(f"\n-- Impostors: {rej}/{len(imp)} = {rej/len(imp):.1%} rejected @ {THRESHOLD}")

# ---------- export failures ----------
if os.path.isdir(FAIL_DIR):
    shutil.rmtree(FAIL_DIR)
nf = 0
for r in records:
    fail, tag = False, ""
    if r['impostor']:
        if r['sim'] >= THRESHOLD: fail, tag = True, f"IMPOSTOR-{r['true']}_as_{r['pred']}"
    elif r['sim'] < THRESHOLD: fail, tag = True, f"{r['true']}_REJECTED"
    elif r['pred'] != r['true']: fail, tag = True, f"{r['true']}_as_{r['pred']}"
    if fail:
        d = os.path.join(FAIL_DIR, r['cond']); os.makedirs(d, exist_ok=True)
        shutil.copy(r['path'], os.path.join(d, f"{tag}__sim{r['sim']:.2f}__{os.path.basename(r['path'])}"))
        nf += 1

os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['cond', 'true', 'pred', 'sim', 'impostor', 'path'])
    w.writeheader(); w.writerows(records)
print(f"\n[SAVED] {nf} failing crops -> {FAIL_DIR}")
print(f"[SAVED] CSV -> {CSV_OUT}")
print("[DONE]")
