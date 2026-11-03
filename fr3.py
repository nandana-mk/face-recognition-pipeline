"""
FR3 — FAR-FACE RECOGNITION, FULLY GPU, MULTI-IMAGE ENROLLMENT
Gallery is built by averaging MANY crops per person from video_dataset/<Person>/train
(robust to distance + lighting), instead of one photo. Detection + embedding on GPU.
Reads a video file (drone footage) or webcam.
"""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import time
from collections import deque, defaultdict

# ==================== CONFIG ====================
# Video file path for drone footage, or 0 for webcam.
import sys
VIDEO_SOURCE = sys.argv[1] if len(sys.argv) > 1 else 0

DETECTOR_NAME = "buffalo_l"
ENROLL_DIR = r"C:\dev\face-recognition-pipeline\video_dataset"   # <Person>/train/*.jpg (112x112 crops)
MAX_ENROLL_PER_PERSON = 200          # cap crops/person at enrollment (evenly sampled)
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Detection
FRAME_SKIP = 3
DOWNSCALE = 1.0
MIN_FACE_SIZE = 12
MIN_DETECTION_SCORE = 0.15
ENABLE_MULTISCALE = True
SCALES = [0.5, 1.0]             # 0.5 = near/huge, 2.0 = far

# Far-face zoom
ZOOM_TIERS = {
    'very_far':   {'threshold': 25, 'context_multiplier': 4.0, 'target_size': 240, 'denoise': False, 'sharpen': True},
    'far':        {'threshold': 40, 'context_multiplier': 3.5, 'target_size': 200, 'denoise': False, 'sharpen': True},
    'medium_far': {'threshold': 60, 'context_multiplier': 2.5, 'target_size': 160, 'denoise': False, 'sharpen': True},
}

# Recognition
SIMILARITY_THRESHOLD = 0.40
EMBED_REFRESH_FRAMES = 8
IOU_THRESHOLD = 0.2

# Quality / display
ENABLE_QUALITY_FILTER = True
MIN_BLUR_THRESHOLD = 50
DISPLAY_MAX_WIDTH = 1600

# ==================== SETUP ====================
os.environ['TMPDIR'] = os.path.join(os.getcwd(), "tmp")
os.makedirs("tmp", exist_ok=True)
print("=" * 70)
print("FR3 — FAR-FACE RECOGNITION (GPU, MULTI-IMAGE ENROLLMENT)")
print("=" * 70)

# ==================== DETECTOR + RECOGNIZER (GPU) ====================
print("[INFO] Initializing detector + recognizer ...")
face_detector = FaceAnalysis(
    name=DETECTOR_NAME,
    allowed_modules=['detection', 'recognition'],
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_detector.prepare(ctx_id=0, det_size=(1280, 1280))
det_model = face_detector.models['detection']
rec_model = face_detector.models['recognition']
print("[INFO] Ready\n")

def get_embedding(face112):
    emb = rec_model.get_feat(face112).flatten().astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)

_clahe_det = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
def enhance_for_detection(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe_det.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    m = img.mean()
    if m < 110:
        g = 1.6 if m < 70 else 1.3
        lut = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)]).astype('uint8')
        img = cv2.LUT(img, lut)
    return img

# ==================== FAR-FACE ZOOM ====================
class AggressiveFarFaceZoom:
    def __init__(self, zoom_tiers):
        self.zoom_tiers = zoom_tiers
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def get_zoom_tier(self, face_size):
        if face_size < self.zoom_tiers['very_far']['threshold']:
            return 'very_far', self.zoom_tiers['very_far']
        elif face_size < self.zoom_tiers['far']['threshold']:
            return 'far', self.zoom_tiers['far']
        elif face_size < self.zoom_tiers['medium_far']['threshold']:
            return 'medium_far', self.zoom_tiers['medium_far']
        return None, None

    def sharpen_adaptive(self, img, intensity='medium'):
        if intensity == 'strong':
            kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]]) / 2
        elif intensity == 'medium':
            kernel = np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]]) / 2
        else:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) / 1
        return cv2.filter2D(img, -1, kernel)

    def enhance_contrast(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def extract_and_enhance(self, frame, bbox, landmarks, face_size):
        tier_name, tier_config = self.get_zoom_tier(face_size)
        if tier_config is None:
            return None, landmarks, {'zoomed': False, 'tier': 'none', 'face_size': face_size}
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        mult = tier_config['context_multiplier']
        ew, eh = (x2 - x1) * mult, (y2 - y1) * mult
        nx1 = int(max(0, cx - ew / 2)); ny1 = int(max(0, cy - eh / 2))
        nx2 = int(min(frame.shape[1], cx + ew / 2)); ny2 = int(min(frame.shape[0], cy + eh / 2))
        extracted = frame[ny1:ny2, nx1:nx2].copy()
        if extracted.size == 0:
            return None, landmarks, {'zoomed': False, 'tier': 'failed', 'face_size': face_size}
        enhanced = extracted
        aw, ah = nx2 - nx1, ny2 - ny1
        if aw > 0 and ah > 0:
            sf = tier_config['target_size'] / max(aw, ah)
            enhanced = cv2.resize(enhanced, (int(aw * sf), int(ah * sf)), interpolation=cv2.INTER_LANCZOS4)
        else:
            sf = 1.0
        enhanced = self.enhance_contrast(enhanced)
        if tier_config['sharpen']:
            enhanced = self.sharpen_adaptive(enhanced, 'strong' if tier_name == 'very_far' else 'medium')
        alm = landmarks.copy()
        alm[:, 0] = (alm[:, 0] - nx1) * sf
        alm[:, 1] = (alm[:, 1] - ny1) * sf
        return enhanced, alm, {'zoomed': True, 'tier': tier_name, 'face_size': face_size}

zoom_extractor = AggressiveFarFaceZoom(ZOOM_TIERS)

def is_face_quality_good(face_img):
    if not ENABLE_QUALITY_FILTER:
        return True
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= MIN_BLUR_THRESHOLD

# ==================== TRACKING ====================
class FaceTrack:
    def __init__(self, track_id, bbox, det_score, frame_num):
        self.track_id = track_id; self.bbox = bbox; self.det_score = det_score
        self.last_seen_frame = frame_num; self.last_embed_frame = -999
        self.names_history = deque(maxlen=10); self.similarities_history = deque(maxlen=10)
        self.stable_name = "Unknown"; self.stable_sim = 0.0; self.frames_seen = 0
        self.zoom_tier = None; self.bbox_history = deque(maxlen=5); self.bbox_history.append(bbox)

    def bbox_changed_significantly(self):
        if len(self.bbox_history) < 2: return False
        c, p = self.bbox_history[-1], self.bbox_history[-2]
        ca = (c[2]-c[0])*(c[3]-c[1]); pa = (p[2]-p[0])*(p[3]-p[1])
        return abs(ca - pa) / (pa + 1e-6) > 0.15

    def needs_new_embedding(self, cf):
        return (cf - self.last_embed_frame >= EMBED_REFRESH_FRAMES) or self.bbox_changed_significantly()

    def update_embedding(self, name, sim, cf, zoom_tier=None):
        self.last_embed_frame = cf; self.zoom_tier = zoom_tier
        self.names_history.append(name); self.similarities_history.append(sim)
        if len(self.names_history) >= 2:
            sc = defaultdict(float)
            for i, (n, s) in enumerate(zip(self.names_history, self.similarities_history)):
                sc[n] += s * ((i + 1) / len(self.names_history))
            self.stable_name = max(sc.items(), key=lambda x: x[1])[0]
            self.stable_sim = float(np.mean(list(self.similarities_history)[-3:]))
        else:
            self.stable_name = name; self.stable_sim = sim

    def update_bbox(self, bbox, det_score, fn):
        self.bbox = bbox; self.det_score = det_score
        self.bbox_history.append(bbox); self.last_seen_frame = fn; self.frames_seen += 1

class TrackManager:
    def __init__(self):
        self.tracks = {}; self.next_id = 0; self.frame_num = 0

    def compute_iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1]); x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        if x2 < x1 or y2 < y1: return 0.0
        inter = (x2-x1)*(y2-y1)
        return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter + 1e-6)

    def update(self, detections):
        self.frame_num += 1; results = []
        for bbox, face, det_score, lm, zi in detections:
            bi, bt = 0, None
            for tid, tr in self.tracks.items():
                iou = self.compute_iou(bbox, tr.bbox)
                if iou > bi and iou > IOU_THRESHOLD: bi, bt = iou, tid
            if bt is not None:
                tr = self.tracks[bt]; tr.update_bbox(bbox, det_score, self.frame_num)
                results.append((tr, face, tr.needs_new_embedding(self.frame_num), zi))
            else:
                tr = FaceTrack(self.next_id, bbox, det_score, self.frame_num)
                self.tracks[self.next_id] = tr; self.next_id += 1
                results.append((tr, face, True, zi))
        for tid in [t for t, tr in self.tracks.items() if self.frame_num - tr.last_seen_frame > 20]:
            del self.tracks[tid]
        return results

track_manager = TrackManager()

# ==================== MULTI-IMAGE ENROLLMENT ====================
print(f"[INFO] Enrolling from {ENROLL_DIR} (averaging crops per person) ...")
gallery_names, _gal = [], []
for person in sorted(os.listdir(ENROLL_DIR)):
    if person.startswith("_"):
        continue
    pdir = os.path.join(ENROLL_DIR, person, "train")
    if not os.path.isdir(pdir):
        continue
    files = [f for f in sorted(os.listdir(pdir)) if f.lower().endswith(VALID_EXT)]
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
    print(f"  [ok] {person:12s}: {len(embs)} crops averaged")

gallery_embs = np.array(_gal, dtype=np.float32)
if len(gallery_names) == 0:
    print("[ERROR] No identities enrolled — check video_dataset/<Person>/train."); exit()
print(f"[INFO] Enrolled {len(gallery_names)}: {', '.join(gallery_names)}\n")

def embed_and_match(face112):
    emb = get_embedding(face112)
    sims = gallery_embs @ emb
    order = np.argsort(sims)[::-1]
    i = int(order[0]); j = int(order[1]) if len(order) > 1 else i
    return gallery_names[i], float(sims[i]), gallery_names[j], float(sims[j])

# ==================== VIDEO / CAMERA ====================
if VIDEO_SOURCE == 0:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"[ERROR] Could not open source: {VIDEO_SOURCE}"); exit()

frame_count = 0; prev_time = time.time()
print("[INFO] Recognition active. Press 'q' to quit.\n")

# ==================== MAIN LOOP ====================
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    display = frame.copy()
    now = time.time()
    detections = []

    if frame_count % FRAME_SKIP == 0:
        small = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE) if DOWNSCALE != 1.0 else frame
        small_det = enhance_for_detection(small)
        all_faces = []
        for scale in (SCALES if ENABLE_MULTISCALE else [1.0]):
            scaled = small_det if scale == 1.0 else cv2.resize(small_det, None, fx=scale, fy=scale)
            try:
                bboxes, kpss = det_model.detect(scaled, max_num=0, metric='default')
            except Exception:
                continue
            if bboxes is None or len(bboxes) == 0:
                continue
            for k in range(bboxes.shape[0]):
                bbox = bboxes[k, :4].copy(); score = float(bboxes[k, 4]); kps = kpss[k].copy()
                if scale != 1.0:
                    bbox, kps = bbox / scale, kps / scale
                all_faces.append({'bbox': bbox, 'kps': kps, 'det_score': score})

        if len(all_faces) > 1:
            all_faces = sorted(all_faces, key=lambda x: x['det_score'], reverse=True)
            kept = []
            for fc in all_faces:
                if not any(track_manager.compute_iou(fc['bbox'], k['bbox']) > 0.5 for k in kept):
                    kept.append(fc)
            all_faces = kept

        for fc in all_faces:
            bbox, kps, det_score = fc['bbox'], fc['kps'], fc['det_score']
            if det_score < MIN_DETECTION_SCORE:
                continue
            x1 = int(bbox[0] / DOWNSCALE); y1 = int(bbox[1] / DOWNSCALE)
            x2 = int(bbox[2] / DOWNSCALE); y2 = int(bbox[3] / DOWNSCALE)
            w, h = x2 - x1, y2 - y1
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            face_size = min(w, h)
            kps_scaled = kps / DOWNSCALE
            bbox_full = (x1, y1, x2, y2)
            er, akps, zi = zoom_extractor.extract_and_enhance(frame, bbox_full, kps_scaled, face_size)
            if er is not None and zi['zoomed']:
                aligned = face_align.norm_crop(er, akps)
            else:
                aligned = face_align.norm_crop(frame, kps_scaled)
                zi = {'zoomed': False, 'tier': 'none'}
            if not is_face_quality_good(aligned):
                continue
            face112 = aligned if aligned.shape[:2] == (112, 112) else cv2.resize(aligned, (112, 112))
            detections.append((bbox_full, face112, det_score, kps_scaled, zi))

    for track, face112, needs_embed, zi in track_manager.update(detections):
        if needs_embed:
            name, sim, name2, sim2 = embed_and_match(face112)
            track.update_embedding(name, sim, track_manager.frame_num, zi.get('tier'))
            x1, y1, x2, y2 = track.bbox
            fs = min(x2 - x1, y2 - y1)
            decision = name if sim >= SIMILARITY_THRESHOLD else f"Unknown#{track.track_id}"
            print(f"[REC] f{track_manager.frame_num:<5d} trk{track.track_id:<2d} {fs:>3d}px "
                  f"{zi.get('tier','none'):<9s} | best={name:<12s} sim={sim:.3f} | "
                  f"2nd={name2:<12s} sim={sim2:.3f} | margin={sim - sim2:+.3f} -> {decision}")

    for tid, track in track_manager.tracks.items():
        if track.frames_seen < 1:
            continue
        label, sim = track.stable_name, track.stable_sim
        if sim >= 0.60:
            color = (0, 255, 0); conf = "HIGH"
        elif sim >= SIMILARITY_THRESHOLD:
            color = (0, 200, 255); conf = "MED"
        else:
            label = f"Unknown #{track.track_id}"; color = (0, 0, 255); conf = "?"
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({conf})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(display, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, f"{min(x2-x1,y2-y1)}px | Sim:{sim:.2f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    fps = 1 / (now - prev_time + 1e-6); prev_time = now
    if frame_count % 30 == 0:
        print(f"[FPS] {fps:5.1f} | frame {frame_count} | tracks: {len(track_manager.tracks)}")
    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if display.shape[1] > DISPLAY_MAX_WIDTH:
        sc = DISPLAY_MAX_WIDTH / display.shape[1]
        display = cv2.resize(display, None, fx=sc, fy=sc)
    cv2.imshow("FR3 - Far-Face Recognition (GPU, multi-enroll)", display)
    if (cv2.waitKey(1) & 0xFF) in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Session complete")
