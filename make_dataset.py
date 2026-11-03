import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2, numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

VIDEO_DIR = r"C:\Users\hticd\OneDrive\Desktop\DJI data\DJI Avata & Mini 4K"
OUT_DIR   = r"C:\dev\face-recognition-pipeline\video_dataset"
FRAME_STRIDE = 15          # sample every 15th frame
MIN_FACE_PX  = 30

# single-person videos -> identity (largest face per frame is auto-labeled)
SINGLE = {
    "Da.MOV": "Dakshina",             # single person -> auto
    "Siddhartha5.MP4": "Siddhartha",  # adds to existing Siddhartha
}
# multi-person videos -> dump ALL faces to _unsorted/ for you to hand-sort
MULTI = [
    "Na, Ni, Ab.mp4", "Na, Ni, Ab 2.mp4",
    "Jaga, Harsh, Allen.mp4", "Jaga, Harsh, Allen 2.mp4",
    "Close-up of Individuals_Sa, Da, Ab, Na, Si, Bk.MOV",
    "From 8m_Sa, Da, Ab, Na, Si, Bk.mp4",
    "In Good Lighting_Sa, Da, Ab, Na, Si, Bk.mp4",
    "Nearer feed_Sa, Da, Ab, Na, Si, Bk.mp4",
    "Walking Towards Drone_Sa, Da, Ab, Na, Si, Bk.mp4",
    "measured distance.mp4",
    "Sa, Da, Ab, Na, Si, Bk_1.MOV", "Sa, Da, Ab, Na, Si, Bk_2.MOV",
    "Sa, Da, Ab, Na, Si, Bk_3.MOV", "Sa, Da, Ab, Na, Si, Bk_4.MOV",
    "Sa, Da, Ab, Na, Si, Bk_5.MOV", "Sa, Da, Ab, Na, Si, Bk_6.MOV",
    "Sa, Da, Ab, Na, Si, Bk_7.MOV", "Sa, Da, Ab, Na, Si, Bk_8.MOV",
    "Sa, Da, Ab, Na, Si, Bk_9.MOV", "Sa, Da, Ab, Na, Si, Bk_10.MOV",
    "Sa, Da, Ab, Na, Si, Bk_11.MOV",
]

app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection'],
                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1280, 1280))
det = app.models['detection']

def frames(path, stride):
    cap = cv2.VideoCapture(path); i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if i % stride == 0: yield i, fr
        i += 1
    cap.release()

def crops(frame):
    bb, kps = det.detect(frame, max_num=0, metric='default')
    out = []
    if bb is None: return out
    for k in range(len(bb)):
        x1, y1, x2, y2 = bb[k, :4]
        px = min(x2 - x1, y2 - y1)
        if px < MIN_FACE_PX: continue
        out.append((face_align.norm_crop(frame, kps[k]), px))
    return out

for vid, label in SINGLE.items():
    p = os.path.join(VIDEO_DIR, vid)
    if not os.path.exists(p): print("missing:", vid); continue
    n = 0
    for fno, fr in frames(p, FRAME_STRIDE):
        cs = crops(fr)
        if not cs: continue
        crop, px = max(cs, key=lambda c: c[1])            # largest face = the subject
        split = "test" if (n % 5 == 0) else "train"       # ~20% test
        d = os.path.join(OUT_DIR, label, split); os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{os.path.splitext(vid)[0]}_{fno:06d}.jpg"), crop); n += 1
    print(f"{label:12s} <- {vid}: {n} crops")

for vid in MULTI:
    p = os.path.join(VIDEO_DIR, vid)
    if not os.path.exists(p): print("missing:", vid); continue
    d = os.path.join(OUT_DIR, "_unsorted", os.path.splitext(vid)[0]); os.makedirs(d, exist_ok=True)
    n = 0
    for fno, fr in frames(p, FRAME_STRIDE):
        for j, (crop, px) in enumerate(crops(fr)):
            cv2.imwrite(os.path.join(d, f"f{fno:06d}_{j}.jpg"), crop); n += 1
    print(f"[manual] {vid}: {n} faces -> {d}")
print("DONE")