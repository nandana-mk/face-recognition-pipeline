"""Ablation: which degradation hurts recognition most?
For sampled clean crops, apply ONE factor at a time (distance/blur/darkness/noise) at
increasing severity, and measure cosine sim of the degraded embedding to the clean one.
Lower sim = more degrading. Uses raw w600k (the deployable recognizer)."""
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
    if img.shape[:2] != (112, 112):
        img = cv2.resize(img, (112, 112))
    e = rec.get_feat(img).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)

# gather clean crops
crops = []
for p in sorted(os.listdir(DS)):
    if p.startswith("_"): continue
    d = os.path.join(DS, p, "train")
    if not os.path.isdir(d): continue
    for f in os.listdir(d):
        if f.lower().endswith(EXT): crops.append(os.path.join(d, f))
np.random.seed(0)
if len(crops) > MAX_CROPS:
    crops = list(np.random.choice(crops, MAX_CROPS, replace=False))
imgs = [cv2.imread(c) for c in crops]
imgs = [cv2.resize(i, (112, 112)) for i in imgs if i is not None]
clean = [emb(i) for i in imgs]
print(f"[INFO] {len(imgs)} clean crops\n")

def mean_sim(degrade):
    s = []
    for im, ce in zip(imgs, clean):
        s.append(float(np.dot(ce, emb(degrade(im)))))
    return np.mean(s)

def far(px):
    def f(im):
        a = cv2.resize(im, (px, px), interpolation=cv2.INTER_AREA)
        return cv2.resize(a, (112, 112), interpolation=cv2.INTER_LINEAR)
    return f
def blur(k):
    return lambda im: cv2.GaussianBlur(im, (k, k), 0)
def dark(f):
    return lambda im: cv2.convertScaleAbs(im, alpha=f, beta=0)
def noise(sig):
    def f(im):
        n = np.random.normal(0, sig, im.shape).astype(np.int16)
        return np.clip(im.astype(np.int16) + n, 0, 255).astype(np.uint8)
    return f

print("=" * 60)
print("DEGRADATION ABLATION  (mean sim to clean; 1.00 = no change)")
print("=" * 60)
print("\nDISTANCE (downscale to N px then back):")
for px in [96, 64, 48, 32, 24, 16]:
    print(f"   {px:>3}px : {mean_sim(far(px)):.3f}")
print("\nBLUR (gaussian kernel):")
for k in [3, 5, 7, 9, 13]:
    print(f"   k={k:>2} : {mean_sim(blur(k)):.3f}")
print("\nDARKNESS (brightness x factor):")
for f in [0.8, 0.6, 0.45, 0.3, 0.2]:
    print(f"   x{f:>4} : {mean_sim(dark(f)):.3f}")
print("\nNOISE (gaussian sigma):")
for s in [6, 12, 20, 30, 45]:
    print(f"   s={s:>2} : {mean_sim(noise(s)):.3f}")
print("\n[DONE] lower value = that factor degrades recognition more.")
