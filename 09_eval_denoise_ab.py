"""A/B: does bilateral denoise help recognition on the labeled _unsorted crops?
Enrolls two galleries (raw + denoised) from video_dataset/<Person>/train, then for every
labeled test crop matches BOTH ways and reports per-condition accuracy RAW vs DENOISE.
Same crops, same threshold -> a clean apples-to-apples comparison."""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2, numpy as np
from collections import defaultdict
from insightface.app import FaceAnalysis

DS  = r"C:\dev\face-recognition-pipeline\video_dataset"
UNS = os.path.join(DS, "_unsorted")
MAX_ENROLL = 200
MAX_TEST_PER_PERSON_PER_COND = 120
THRESHOLD = 0.40
EXT = ('.jpg', '.jpeg', '.png', '.bmp')

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
rec = app.models['recognition']

def _emb(img):
    if img.shape[:2] != (112, 112): img = cv2.resize(img, (112, 112))
    e = rec.get_feat(img).flatten().astype(np.float32)
    return e / (np.linalg.norm(e) + 1e-9)
def emb_raw(img): return _emb(img)
def emb_den(img): return _emb(cv2.bilateralFilter(img, 5, 50, 50))

# ---- enroll two galleries ----
names, gr, gd = [], [], []
for p in sorted(os.listdir(DS)):
    if p.startswith("_"): continue
    d = os.path.join(DS, p, "train")
    if not os.path.isdir(d): continue
    files = [f for f in sorted(os.listdir(d)) if f.lower().endswith(EXT)]
    if len(files) > MAX_ENROLL:
        step = len(files) / MAX_ENROLL
        files = [files[int(i*step)] for i in range(MAX_ENROLL)]
    er, ed = [], []
    for f in files:
        im = cv2.imread(os.path.join(d, f))
        if im is None: continue
        er.append(emb_raw(im)); ed.append(emb_den(im))
    if er:
        vr = np.mean(er, 0); vr /= np.linalg.norm(vr)+1e-9
        vd = np.mean(ed, 0); vd /= np.linalg.norm(vd)+1e-9
        names.append(p); gr.append(vr.astype(np.float32)); gd.append(vd.astype(np.float32))
G_raw = np.array(gr, np.float32); G_den = np.array(gd, np.float32)
print(f"[INFO] gallery: {', '.join(names)}\n")

def match(v, G): i = int(np.argmax(G @ v)); return names[i], float((G @ v)[i])

# ---- evaluate labeled test crops ----
recs = []
for cond in sorted(os.listdir(UNS)):
    cpath = os.path.join(UNS, cond)
    if not os.path.isdir(cpath): continue
    for person in sorted(os.listdir(cpath)):
        if person.startswith("_"): continue
        pp = os.path.join(cpath, person)
        if not os.path.isdir(pp): continue
        files = [f for f in os.listdir(pp) if f.lower().endswith(EXT)]
        if len(files) > MAX_TEST_PER_PERSON_PER_COND:
            files = list(np.random.RandomState(0).choice(files, MAX_TEST_PER_PERSON_PER_COND, replace=False))
        for f in files:
            im = cv2.imread(os.path.join(pp, f))
            if im is None: continue
            pr, sr = match(emb_raw(im), G_raw)
            pdn, sdn = match(emb_den(im), G_den)
            recs.append({'cond': cond, 'true': person, 'impostor': person not in names,
                         'pr': pr, 'sr': sr, 'pd': pdn, 'sd': sdn})
print(f"[INFO] {len(recs)} test crops\n")

def acc(sub, key_p, key_s):
    ok = sum(1 for r in sub if r[key_s] >= THRESHOLD and r[key_p] == r['true'])
    return ok / len(sub) if sub else 0.0

gen = [r for r in recs if not r['impostor']]
print("=" * 64)
print(f"BILATERAL DENOISE A/B  (genuine accuracy @ {THRESHOLD})")
print("=" * 64)
print(f"  OVERALL   raw {acc(gen,'pr','sr'):6.1%}  |  denoise {acc(gen,'pd','sd'):6.1%}   (n={len(gen)})")
print("\n  per-condition:            raw    denoise   delta")
by = defaultdict(list)
for r in gen: by[r['cond']].append(r)
for c in sorted(by):
    b = by[c]; ar = acc(b,'pr','sr'); ad = acc(b,'pd','sd')
    print(f"  {c[:26]:26s} {ar:6.1%}  {ad:6.1%}  {ad-ar:+6.1%}  (n={len(b)})")

imp = [r for r in recs if r['impostor']]
if imp:
    rej_r = sum(1 for r in imp if r['sr'] < THRESHOLD)/len(imp)
    rej_d = sum(1 for r in imp if r['sd'] < THRESHOLD)/len(imp)
    print(f"\n  impostor reject   raw {rej_r:6.1%}  |  denoise {rej_d:6.1%}   (n={len(imp)})")
print("\n[DONE] positive delta = denoise helped that condition.")
