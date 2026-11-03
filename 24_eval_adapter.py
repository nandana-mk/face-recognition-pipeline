"""
24_eval_adapter.py — STEP 3: the GO / NO-GO gates. Run BEFORE deploying any adapter.

    python 24_eval_adapter.py "C:\\Users\\hticd\\scikit_learn_data\\lfw_home\\lfw_funneled"

This is the step that was skipped last time and let a bad adapter ship. It answers three
questions, using SINGLE-PORTRAIT enrollment (matching the real constraint: one reference photo):

  GATE 1  RECALL      DroneFace recall by face-pixel band, with vs without the adapter.
                      MUST IMPROVE in the 45-90 px band (where signal exists but is misaligned).
                      Below ~45 px nothing can help - the information was never captured.

  GATE 2  PRECISION   Stranger rejection. LFW faces (people definitely NOT on the watchlist) are
                      degraded the same way and scored against the DroneFace gallery.
                      False-accept rate MUST NOT rise. This is the gate that matters most.

  GATE 3  NO REGRESSION   Clean close-range crops (>=120 px) top-1 accuracy MUST NOT fall.
                      An adapter that fixes far faces by breaking near ones is a bad trade.

Verdict: ALL THREE must pass. If any fails -> DISCARD the adapter. Do not retune against these
numbers until they pass; that is overfitting the validation set and is how the last one regressed.
"""
import os, sys
_TORCH_LIB = r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib"
os.environ["PATH"] = _TORCH_LIB + os.pathsep + os.environ.get("PATH", "")
os.add_dll_directory(_TORCH_LIB)
import cv2, numpy as np, glob
import torch, torch.nn as nn, torch.nn.functional as F
from insightface.app import FaceAnalysis
from insightface.utils import face_align

BASE   = r"C:\dev\train_val\datasets\raw\DnHFaces\open_data_set"
FACES  = os.path.join(BASE, "photos_all_faces")
PORT   = os.path.join(BASE, "portraits")
TRIO_C = os.path.join(BASE, "trio_cam")
TRIO_G = os.path.join(BASE, "trio_gp")
ADAPTER = r"C:\dev\train_val\models\adapter_deg.pt"
THRESHOLD = 0.40
N_IMPOSTORS = 1500
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')
SEED = 1

rng = np.random.default_rng(SEED)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
det, rec = app.models['detection'], app.models['recognition']

# ---------------- adapter ----------------
class Adapter(nn.Module):
    def __init__(self, dim=512, h=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True),
                                 nn.Dropout(0.1), nn.Linear(h, dim))
    def forward(self, x):
        return F.normalize(x + self.net(x), dim=1)

ck = torch.load(ADAPTER, map_location=dev)
model = Adapter(ck.get('dim', 512), ck.get('hidden', 1024)).to(dev)
model.load_state_dict(ck['state_dict']); model.eval()
print(f"[INFO] adapter loaded from {ADAPTER}")

def adapt(E):
    with torch.no_grad():
        out = model(torch.from_numpy(E).to(dev)).cpu().numpy()
    return out.astype(np.float32)

def embed_batch(crops):
    f = rec.get_feat(crops)
    f = np.asarray(f, np.float32).reshape(len(crops), -1)
    return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)

def align_largest(im, min_px=0):
    bb, kps = det.detect(im, max_num=0, metric='default')
    if bb is None or len(bb) == 0: return None
    j = int(np.argmax((bb[:, 2]-bb[:, 0]) * (bb[:, 3]-bb[:, 1])))
    if min(bb[j, 2]-bb[j, 0], bb[j, 3]-bb[j, 1]) < min_px: return None
    return face_align.norm_crop(im, kps[j])

def find_frontal(s):
    for folder, pre in [(PORT, f"{s}_"), (TRIO_C, f"{s}_cam_0_porf"), (TRIO_G, f"{s}_gp_0_porf")]:
        if not os.path.isdir(folder): continue
        for f in sorted(os.listdir(folder)):
            if f.lower().startswith(pre) and f.lower().endswith(IMG_EXT):
                return os.path.join(folder, f)
    return None

# ---------------- gallery: ONE portrait per subject ----------------
subjects = sorted({f.split('_')[0] for f in os.listdir(FACES) if f.lower().endswith(IMG_EXT)})
names, gcrops = [], []
for s in subjects:
    fp = find_frontal(s)
    if not fp: continue
    im = cv2.imread(fp)
    if im is None: continue
    c = align_largest(im) if im.shape[:2] != (112, 112) else im
    if c is None: c = cv2.resize(im, (112, 112))
    names.append(s); gcrops.append(c)
G_raw = embed_batch(gcrops); G_ad = adapt(G_raw)
print(f"[INFO] enrolled {len(names)} subjects from ONE portrait each: {names}\n")

# ---------------- probes: all DroneFace crops ----------------
crops, sids, pxs = [], [], []
for f in os.listdir(FACES):
    if not f.lower().endswith(IMG_EXT): continue
    p = f.split('_')
    if len(p) < 5 or p[1] != 'gp' or p[0] not in names: continue
    im = cv2.imread(os.path.join(FACES, f))
    if im is None: continue
    pxs.append(min(im.shape[0], im.shape[1])); sids.append(p[0])
    crops.append(cv2.resize(im, (112, 112)) if im.shape[:2] != (112, 112) else im)
P_raw = np.concatenate([embed_batch(crops[i:i+256]) for i in range(0, len(crops), 256)], 0)
P_ad = adapt(P_raw)
pxs = np.array(pxs); sids = np.array(sids)
print(f"[INFO] {len(crops)} DroneFace probe crops\n")

def recall_by_band(P, G):
    S = P @ G.T
    top = S.argmax(1); best = S.max(1)
    ok = (np.array(names)[top] == sids) & (best >= THRESHOLD)
    out = {}
    for lo, hi, lab in [(0,35,'<35'), (35,45,'35-45'), (45,55,'45-55'), (55,70,'55-70'),
                        (70,90,'70-90'), (90,120,'90-120'), (120,10**6,'120+')]:
        m = (pxs >= lo) & (pxs < hi)
        out[lab] = (ok[m].mean() if m.sum() else np.nan, int(m.sum()))
    return out, ok

r_raw, ok_raw = recall_by_band(P_raw, G_raw)
r_ad,  ok_ad  = recall_by_band(P_ad,  G_ad)

print("=" * 72)
print("GATE 1 - RECALL by face pixel band (single-portrait enrollment)")
print("=" * 72)
print(f"  {'band':>8} {'n':>5} {'baseline':>10} {'adapter':>10} {'delta':>8}")
band_ok = True
for lab in ['<35', '35-45', '45-55', '55-70', '70-90', '90-120', '120+']:
    b, n = r_raw[lab]; a, _ = r_ad[lab]
    if np.isnan(b): continue
    d = a - b
    mark = ""
    if lab in ('45-55', '55-70', '70-90'):
        if d < 0: band_ok = False; mark = "  <-- REGRESSED"
        elif d > 0.02: mark = "  <-- improved"
    print(f"  {lab:>8} {n:>5} {b:>9.1%} {a:>9.1%} {d:>+8.1%}{mark}")
tgt = [(45,55),(55,70),(70,90)]
m_t = np.zeros(len(pxs), bool)
for lo, hi in tgt: m_t |= (pxs >= lo) & (pxs < hi)
g1_base, g1_ad = ok_raw[m_t].mean(), ok_ad[m_t].mean()
print(f"\n  TARGET BAND 45-90px:  {g1_base:.1%} -> {g1_ad:.1%}  ({g1_ad-g1_base:+.1%})")
GATE1 = (g1_ad > g1_base) and band_ok

# ---------------- GATE 2: strangers ----------------
lfw = sys.argv[1] if len(sys.argv) > 1 else None
GATE2 = None
if lfw and os.path.isdir(lfw):
    files = [p for p in glob.glob(os.path.join(lfw, '**', '*'), recursive=True)
             if p.lower().endswith(IMG_EXT)]
    rng.shuffle(files); files = files[:N_IMPOSTORS * 2]
    icrops = []
    for p in files:
        if len(icrops) >= N_IMPOSTORS: break
        im = cv2.imread(p)
        if im is None: continue
        c = align_largest(im, min_px=50)
        if c is None: continue
        k = int(rng.integers(30, 81))                       # degrade like a drone probe
        c = cv2.resize(cv2.resize(c, (k, k), interpolation=cv2.INTER_AREA), (112, 112))
        if rng.random() < 0.7:
            c = np.clip(c.astype(np.float32) + rng.normal(0, rng.uniform(2, 15), c.shape),
                        0, 255).astype(np.uint8)
        icrops.append(c)
    if icrops:
        I_raw = np.concatenate([embed_batch(icrops[i:i+256]) for i in range(0, len(icrops), 256)], 0)
        I_ad = adapt(I_raw)
        fa_raw = (I_raw @ G_raw.T).max(1)
        fa_ad = (I_ad @ G_ad.T).max(1)
        r_fa, a_fa = (fa_raw >= THRESHOLD).mean(), (fa_ad >= THRESHOLD).mean()
        print("\n" + "=" * 72)
        print(f"GATE 2 - STRANGER REJECTION ({len(icrops)} LFW faces, degraded, not enrolled)")
        print("=" * 72)
        print(f"  false-accept rate   baseline {r_fa:.2%}   adapter {a_fa:.2%}   ({a_fa-r_fa:+.2%})")
        print(f"  rejection rate      baseline {1-r_fa:.2%}   adapter {1-a_fa:.2%}")
        print(f"  mean max-sim        baseline {fa_raw.mean():.3f}   adapter {fa_ad.mean():.3f}")
        GATE2 = (a_fa <= r_fa + 0.005) and (a_fa <= 0.01)
else:
    print("\n[GATE 2 SKIPPED] pass the LFW folder as an argument to test stranger rejection.")

# ---------------- GATE 3: no regression on clean ----------------
m_c = pxs >= 120
c_base, c_ad = ok_raw[m_c].mean(), ok_ad[m_c].mean()
print("\n" + "=" * 72)
print("GATE 3 - NO REGRESSION on clean close-range crops (>=120 px)")
print("=" * 72)
print(f"  accuracy   baseline {c_base:.1%}   adapter {c_ad:.1%}   ({c_ad-c_base:+.1%})   n={m_c.sum()}")
GATE3 = c_ad >= c_base - 0.01

# ---------------- verdict ----------------
print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
def show(n, v):
    print(f"  {n:<34} {'PASS' if v else ('SKIPPED' if v is None else 'FAIL')}")
show("GATE 1  recall 45-90px improves", GATE1)
show("GATE 2  stranger rejection held", GATE2)
show("GATE 3  no clean-crop regression", GATE3)
gates = [GATE1, GATE3] + ([GATE2] if GATE2 is not None else [])
if all(gates):
    print("\n  >>> GO. Apply the adapter to ALL embeddings (gallery and probe), then RE-RUN")
    print("      15_calibrate_decision.py - the similarity distribution has shifted, so your")
    print("      0.40 threshold no longer means what it meant.")
else:
    print("\n  >>> NO-GO. Discard this adapter.")
    print("      Do NOT retune against these numbers until they pass - that overfits the")
    print("      validation set. Fix the cause instead: more training identities (LFW is thin;")
    print("      CASIA-WebFace is the proper set), or raise LAMBDA if clean crops regressed.")
