"""Measure how much denoising/sharpening RECOVERS recognition on noisy & far faces.
Degrade clean crops -> apply candidate fixes -> measure cosine sim to the clean embedding.
Higher recovered sim = the fix helps. Targets the #1 degrader (noise) and realistic far."""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2, numpy as np
from insightface.app import FaceAnalysis

DS = r"C:\dev\face-recognition-pipeline\video_dataset"
MAX_CROPS = 150
EXT = ('.jpg', '.jpeg', '.png', '.bmp')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
rec = app.models['recognition']
def emb(img):
    if img.shape[:2] != (112, 112): img = cv2.resize(img, (112, 112))
    e = rec.get_feat(img).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

crops = []
for p in sorted(os.listdir(DS)):
    if p.startswith("_"): continue
    d = os.path.join(DS, p, "train")
    if os.path.isdir(d):
        crops += [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(EXT)]
np.random.seed(0)
if len(crops) > MAX_CROPS: crops = list(np.random.choice(crops, MAX_CROPS, replace=False))
imgs = [cv2.resize(cv2.imread(c), (112, 112)) for c in crops if cv2.imread(c) is not None]
clean = [emb(i) for i in imgs]
print(f"[INFO] {len(imgs)} clean crops\n")

# ---- degradations (simulate real far/noisy drone faces) ----
def add_noise(img, sig):
    n = np.random.normal(0, sig, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + n, 0, 255).astype(np.uint8)
def deg_noise(img):      # noise only (sigma 20)
    return add_noise(img, 20)
def deg_far(img):        # far: downscale 32px + noise 15 + slight blur
    a = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
    a = cv2.resize(a, (112, 112), interpolation=cv2.INTER_LINEAR)
    a = add_noise(a, 15)
    return cv2.GaussianBlur(a, (3, 3), 0)

# ---- candidate fixes ----
def f_none(x): return x
def f_bilateral(x): return cv2.bilateralFilter(x, 5, 50, 50)
def f_nlm(x): return cv2.fastNlMeansDenoisingColored(x, None, 7, 7, 7, 21)
def f_nlm_sharp(x):
    y = cv2.fastNlMeansDenoisingColored(x, None, 7, 7, 7, 21)
    return cv2.addWeighted(y, 1.5, cv2.GaussianBlur(y, (0, 0), 2), -0.5, 0)
def f_bilateral_clahe(x):
    y = cv2.bilateralFilter(x, 5, 50, 50)
    lab = cv2.cvtColor(y, cv2.COLOR_BGR2LAB); l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

FIXES = [("none/baseline", f_none), ("bilateral", f_bilateral), ("NLM", f_nlm),
         ("NLM+sharpen", f_nlm_sharp), ("bilateral+CLAHE", f_bilateral_clahe)]

def run(deg, title):
    print(f"== {title} ==  (mean sim to clean; higher = better)")
    for name, fn in FIXES:
        s = np.mean([float(np.dot(ce, emb(fn(deg(im))))) for im, ce in zip(imgs, clean)])
        print(f"   {name:18s}: {s:.3f}")
    print()

print("=" * 55)
run(deg_noise, "NOISE only (sigma 20)")
run(deg_far,   "REALISTIC FAR (32px + noise15 + blur)")
print("[DONE] compare each fix vs baseline to see recovery.")
