"""
enroll_preview.py — audit the Known_faces references used for recognition.
For each photo: detect the largest face, align to 112x112 (exactly what enrollment uses),
and report quality + confusability. Saves:
  reports/enroll_preview/<name>.jpg        — the aligned crop actually enrolled
  reports/enroll_preview/_contact_sheet.jpg — all references in one labeled grid
Flags weak refs: blurry, small face, or too-similar to another person (confusion risk).
"""
import os
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

KNOWN_DIR = r"C:\dev\face-recognition-pipeline\Known_faces"
OUT = r"C:\dev\train_val\reports\enroll_preview"
EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
BLUR_MIN = 80          # Laplacian variance below this = blurry reference
PX_MIN = 60            # detected face smaller than this = weak/low-detail reference
CONFUSE_MAX = 0.45     # cosine to nearest OTHER person above this = confusion risk
os.makedirs(OUT, exist_ok=True)

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

refs = []   # (name, crop112, emb, px, det_score, blur)
for f in sorted(os.listdir(KNOWN_DIR)):
    if not f.lower().endswith(EXT):
        continue
    name = os.path.splitext(f)[0]
    im = cv2.imread(os.path.join(KNOWN_DIR, f))
    if im is None:
        print(f"[skip] cannot read {f}"); continue
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0:
        print(f"[WARN] NO FACE detected in {f} -> this person can't be enrolled!"); continue
    k = int(np.argmax((bb[:, 2]-bb[:, 0])*(bb[:, 3]-bb[:, 1])))
    px = int(min(bb[k, 2]-bb[k, 0], bb[k, 3]-bb[k, 1]))
    crop = face_align.norm_crop(im, kps[k])
    e = rec.get_feat(crop).flatten().astype(np.float32); e /= (np.linalg.norm(e)+1e-9)
    blur = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    refs.append([name, crop, e, px, float(bb[k, 4]), float(blur)])
    cv2.imwrite(os.path.join(OUT, f"{name}.jpg"), crop)

if not refs:
    print("[ERROR] no faces enrolled."); raise SystemExit

# cross-similarity: nearest OTHER person for each
E = np.array([r[2] for r in refs])
S = E @ E.T
np.fill_diagonal(S, -1)
names = [r[0] for r in refs]

print("\n" + "=" * 74)
print("ENROLLMENT REFERENCE AUDIT  (Known_faces)")
print("=" * 74)
print(f"  {'name':12s} {'facePx':>6} {'detScore':>8} {'blur':>7} {'nearest-other':>16} {'sim':>5}  flags")
flagged = []
for i, r in enumerate(refs):
    j = int(np.argmax(S[i])); nn, ns = names[j], float(S[i, j])
    fl = []
    if r[5] < BLUR_MIN: fl.append("BLURRY")
    if r[3] < PX_MIN:   fl.append("SMALL")
    if ns > CONFUSE_MAX: fl.append(f"CONFUSABLE~{nn}")
    if fl: flagged.append(r[0])
    print(f"  {r[0]:12s} {r[3]:>6} {r[4]:>8.2f} {r[5]:>7.0f} {nn:>16} {ns:>5.2f}  {' '.join(fl)}")

# contact sheet
CELL, PAD, COLS = 170, 10, 5
rows = (len(refs) + COLS - 1) // COLS
sheet = np.full((rows*(CELL+34)+PAD, COLS*(CELL+PAD)+PAD, 3), 30, np.uint8)
for idx, r in enumerate(refs):
    rr, cc = idx // COLS, idx % COLS
    y, x = PAD + rr*(CELL+34), PAD + cc*(CELL+PAD)
    thumb = cv2.resize(r[1], (CELL, CELL))
    border = (0, 0, 220) if r[0] in flagged else (0, 200, 0)
    cv2.copyMakeBorder(thumb, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=border)
    sheet[y:y+CELL, x:x+CELL] = thumb
    cv2.rectangle(sheet, (x, y), (x+CELL, y+CELL), border, 2)
    cv2.putText(sheet, f"{r[0]}", (x, y+CELL+16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(sheet, f"blur{r[5]:.0f} {r[3]}px", (x, y+CELL+30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
cv2.imwrite(os.path.join(OUT, "_contact_sheet.jpg"), sheet)

print(f"\n[SAVED] individual crops + _contact_sheet.jpg -> {OUT}")
print(f"[FLAGGED] {', '.join(flagged) if flagged else 'none'}  (red border in the sheet)")
print("Green border = good reference · Red = blurry / small / confusable -> replace that photo.")
