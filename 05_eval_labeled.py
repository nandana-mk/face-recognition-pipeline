"""
05_eval_labeled.py — CONDITION-AWARE EVALUATION on hand-labeled crops.
Reads test crops from  video_dataset/_unsorted/<video>/<Person>/*.jpg
(the <video> name = the condition: far / dark / 8m / good-light / ...).
Enrolls a multi-image gallery from video_dataset/<Person>/train, then reports:
  - overall + PER-CONDITION accuracy, threshold sweep + PR-AUC
  - false-accept on people NOT in the gallery (Bk / Jaga / Harsh = impostors)
  - EXPORTS every failing crop to reports/failures/<condition>/ so you see the exact pictures.
GPU (insightface w600k).
"""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2, numpy as np, shutil, csv
from collections import defaultdict
from insightface.app import FaceAnalysis

DS       = r"C:\dev\face-recognition-pipeline\video_dataset"
UNS      = os.path.join(DS, "_unsorted")
FAIL_DIR = r"C:\dev\train_val\reports\failures"
CSV_OUT  = r"C:\dev\train_val\reports\labeled_eval.csv"
MAX_ENROLL_PER_PERSON = 200
THRESHOLD = 0.45                       # accept cutoff for the per-condition tables
EXT = ('.jpg', '.jpeg', '.png', '.bmp')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
rec = app.models['recognition']

def emb(img):
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112))
    e = rec.get_feat(img).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

# ---------- enroll (multi-image) ----------
print("[INFO] enrolling gallery from train crops ...")
gallery_names, _gal = [], []
for person in sorted(os.listdir(DS)):
    if person.startswith("_"):
        continue
    pdir = os.path.join(DS, person, "train")
    if not os.path.isdir(pdir):
        continue
    files = [f for f in sorted(os.listdir(pdir)) if f.lower().endswith(EXT)]
    if len(files) > MAX_ENROLL_PER_PERSON:
        step = len(files) / MAX_ENROLL_PER_PERSON
        files = [files[int(i * step)] for i in range(MAX_ENROLL_PER_PERSON)]
    es = []
    for f in files:
        im = cv2.imread(os.path.join(pdir, f))
        if im is not None:
            es.append(emb(im))
    if es:
        v = np.mean(es, axis=0); v /= (np.linalg.norm(v) + 1e-9)
        gallery_names.append(person); _gal.append(v.astype(np.float32))
        print(f"  [ok] {person:12s}: {len(es)}")
gallery_embs = np.array(_gal, dtype=np.float32)
if not gallery_names:
    print("[ERROR] empty gallery."); raise SystemExit
print(f"[INFO] gallery: {', '.join(gallery_names)}\n")

# ---------- run labeled test crops ----------
records = []
if not os.path.isdir(UNS):
    print(f"[ERROR] no _unsorted at {UNS}"); raise SystemExit
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
    print("[INFO] no labeled crops found yet — fill the person subfolders first."); raise SystemExit
print(f"[INFO] {len(records)} labeled test crops\n")

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

ts = [round(x, 2) for x in np.arange(0.20, 0.85, 0.02)]
rows = [sweep(t) for t in ts]
pairs = sorted((r['R'], r['P']) for r in rows)
pr_auc = float(np.trapezoid([p for _, p in pairs], [r for r, _ in pairs]))
best_f1 = max(rows, key=lambda r: r['F1'])

print("=" * 70)
print("LABELED EVALUATION")
print("=" * 70)
print(f"PR-AUC: {pr_auc:.4f}   Best-F1 @ t={best_f1['t']:.2f}: "
      f"P={best_f1['P']:.3f} R={best_f1['R']:.3f} F1={best_f1['F1']:.3f}")

m = sweep(THRESHOLD)
_tot = m['TP'] + m['FP'] + m['FN'] + m['TN']
_acc = (m['TP'] + m['TN']) / (_tot + 1e-9)
print(f"\n@ operating threshold {THRESHOLD}:")
print(f"  Accuracy  {_acc:.3f}    Precision {m['P']:.3f}    Recall {m['R']:.3f}    F1 {m['F1']:.3f}")
print(f"  TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  TN={m['TN']}   (n={_tot})")

print(f"\n-- Per-condition accuracy (genuine, threshold {THRESHOLD}) --")
by_cond = defaultdict(list)
for r in records:
    if not r['impostor']:
        by_cond[r['cond']].append(r)
for cond in sorted(by_cond):
    b = by_cond[cond]
    ok = sum(1 for r in b if r['sim'] >= THRESHOLD and r['pred'] == r['true'])
    print(f"  {cond[:38]:38s} {ok}/{len(b)} = {ok/len(b):6.1%}  (mean sim {np.mean([r['sim'] for r in b]):.3f})")

print(f"\n-- Per-person accuracy (genuine, threshold {THRESHOLD}) --")
by_p = defaultdict(list)
for r in records:
    if not r['impostor']:
        by_p[r['true']].append(r)
for p in sorted(by_p):
    b = by_p[p]
    ok = sum(1 for r in b if r['sim'] >= THRESHOLD and r['pred'] == r['true'])
    print(f"  {p:12s} {ok}/{len(b)} = {ok/len(b):6.1%}")

imp = [r for r in records if r['impostor']]
if imp:
    rej = sum(1 for r in imp if r['sim'] < THRESHOLD)
    print(f"\n-- Impostors (not enrolled): {rej}/{len(imp)} = {rej/len(imp):.1%} correctly rejected @ {THRESHOLD}")

# ---------- export failing crops ----------
if os.path.isdir(FAIL_DIR):
    shutil.rmtree(FAIL_DIR)
nfail = 0
for r in records:
    fail = False; tag = ""
    if r['impostor']:
        if r['sim'] >= THRESHOLD:
            fail = True; tag = f"IMPOSTOR-{r['true']}_as_{r['pred']}"
    else:
        if r['sim'] < THRESHOLD:
            fail = True; tag = f"{r['true']}_REJECTED"
        elif r['pred'] != r['true']:
            fail = True; tag = f"{r['true']}_as_{r['pred']}"
    if fail:
        d = os.path.join(FAIL_DIR, r['cond']); os.makedirs(d, exist_ok=True)
        shutil.copy(r['path'], os.path.join(d, f"{tag}__sim{r['sim']:.2f}__{os.path.basename(r['path'])}"))
        nfail += 1

os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
with open(CSV_OUT, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['cond', 'true', 'pred', 'sim', 'impostor', 'path'])
    w.writeheader(); w.writerows(records)

print(f"\n[SAVED] {nfail} failing crops -> {FAIL_DIR}\\<condition>\\")
print(f"[SAVED] per-crop CSV -> {CSV_OUT}")
print("[DONE]")
