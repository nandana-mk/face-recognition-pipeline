"""
FR2 — SCRFD FAR-FACE RECOGNITION on DRONE VIDEO, FULLY GPU
Detection (SCRFD) AND embedding (insightface w600k) both run on onnxruntime-GPU.
Reads from a video file (drone footage) or webcam. No DeepFace / TensorFlow / adapter.
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
# Set VIDEO_SOURCE to a file path for drone footage, or 0 for webcam.
VIDEO_SOURCE = r"C:\Users\hticd\OneDrive\Desktop\DJI data\DJI Avata & Mini 4K\Dark Na,Ab,Si,Ni.mp4"

DETECTOR_NAME = "buffalo_l"                 # w600k_r50 recognizer
KNOWN_FACES_DIR = r"C:\dev\face-recognition-pipeline\Known_faces"
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Detection settings
FRAME_SKIP = 3
DOWNSCALE = 1.0
MIN_FACE_SIZE = 12
MIN_DETECTION_SCORE = 0.15

# Multi-scale detection (ON for far faces; heavier but better range)
ENABLE_MULTISCALE = True
SCALES = [0.5,1.0]

# Far-face zoom
ENABLE_FAR_FACE_ZOOM = True
ZOOM_TIERS = {
    'very_far':   {'threshold': 25, 'context_multiplier': 4.0, 'target_size': 240, 'denoise': False, 'sharpen': True},
    'far':        {'threshold': 40, 'context_multiplier': 3.5, 'target_size': 200, 'denoise': False, 'sharpen': True},
    'medium_far': {'threshold': 60, 'context_multiplier': 2.5, 'target_size': 160, 'denoise': False, 'sharpen': True},
}

# Recognition
SIMILARITY_THRESHOLD = 0.38     # starting point; calibrate from [REC] log

# Track-based caching
EMBED_REFRESH_FRAMES = 8
IOU_THRESHOLD = 0.2

# Quality filtering
ENABLE_QUALITY_FILTER = True
MIN_BLUR_THRESHOLD = 50

# Display (shrink huge 4K frames to fit the screen)
DISPLAY_MAX_WIDTH = 1600

# ==================== SETUP ====================
os.environ['TMPDIR'] = os.path.join(os.getcwd(), "tmp")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs("tmp", exist_ok=True)

print("=" * 70)
print("FR2 — FAR-FACE RECOGNITION ON VIDEO (FULLY GPU)")
print("=" * 70)

# ==================== DETECTOR + RECOGNIZER (GPU) ====================
print("[INFO] Initializing detector + recognizer ...")
face_detector = FaceAnalysis(
    name=DETECTOR_NAME,
    allowed_modules=['detection', 'recognition'],   # skip landmark/genderage models (faster)
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_detector.prepare(ctx_id=0, det_size=(960, 960))
det_model = face_detector.models['detection']
rec_model = face_detector.models['recognition']
print("[INFO] Ready\n")

def get_embedding(face112):
    """112x112 BGR aligned crop -> L2-normalized 512-d embedding (GPU)."""
    emb = rec_model.get_feat(face112).flatten().astype(np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)

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

    def denoise_strong(self, img):
        return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)

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
        face_w, face_h = x2 - x1, y2 - y1
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        multiplier = tier_config['context_multiplier']
        expanded_w, expanded_h = face_w * multiplier, face_h * multiplier

        new_x1 = int(max(0, center_x - expanded_w / 2))
        new_y1 = int(max(0, center_y - expanded_h / 2))
        new_x2 = int(min(frame.shape[1], center_x + expanded_w / 2))
        new_y2 = int(min(frame.shape[0], center_y + expanded_h / 2))

        extracted = frame[new_y1:new_y2, new_x1:new_x2].copy()
        if extracted.size == 0:
            return None, landmarks, {'zoomed': False, 'tier': 'failed', 'face_size': face_size}

        enhanced = extracted.copy()
        if tier_config['denoise']:
            if tier_name == 'very_far':
                enhanced = self.denoise_strong(enhanced)
            else:
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

        actual_w, actual_h = new_x2 - new_x1, new_y2 - new_y1
        if actual_w > 0 and actual_h > 0:
            target = tier_config['target_size']
            scale_factor = target / max(actual_w, actual_h)
            new_w, new_h = int(actual_w * scale_factor), int(actual_h * scale_factor)
            enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        else:
            scale_factor = 1.0

        enhanced = self.enhance_contrast(enhanced)
        if tier_config['sharpen']:
            if tier_name == 'very_far':
                enhanced = self.sharpen_adaptive(enhanced, 'strong')
            elif tier_name == 'far':
                enhanced = self.sharpen_adaptive(enhanced, 'medium')
            else:
                enhanced = self.sharpen_adaptive(enhanced, 'light')

        adjusted_landmarks = landmarks.copy()
        adjusted_landmarks[:, 0] = (adjusted_landmarks[:, 0] - new_x1) * scale_factor
        adjusted_landmarks[:, 1] = (adjusted_landmarks[:, 1] - new_y1) * scale_factor

        zoom_info = {'zoomed': True, 'tier': tier_name, 'face_size': face_size, 'scale_factor': scale_factor}
        return enhanced, adjusted_landmarks, zoom_info

zoom_extractor = AggressiveFarFaceZoom(ZOOM_TIERS)

# ==================== QUALITY FILTER ====================
def is_face_quality_good(face_img):
    if not ENABLE_QUALITY_FILTER:
        return True
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= MIN_BLUR_THRESHOLD

# ==================== FACE TRACK ====================
class FaceTrack:
    def __init__(self, track_id, bbox, det_score, frame_num):
        self.track_id = track_id
        self.bbox = bbox
        self.det_score = det_score
        self.last_seen_frame = frame_num
        self.last_embed_frame = -999
        self.names_history = deque(maxlen=10)
        self.similarities_history = deque(maxlen=10)
        self.stable_name = "Unknown"
        self.stable_sim = 0.0
        self.frames_seen = 0
        self.zoom_tier = None
        self.bbox_history = deque(maxlen=5)
        self.bbox_history.append(bbox)

    def bbox_changed_significantly(self):
        if len(self.bbox_history) < 2:
            return False
        cur, prev = self.bbox_history[-1], self.bbox_history[-2]
        ca = (cur[2] - cur[0]) * (cur[3] - cur[1])
        pa = (prev[2] - prev[0]) * (prev[3] - prev[1])
        return abs(ca - pa) / (pa + 1e-6) > 0.15

    def needs_new_embedding(self, current_frame):
        if current_frame - self.last_embed_frame >= EMBED_REFRESH_FRAMES:
            return True
        return self.bbox_changed_significantly()

    def update_embedding(self, name, similarity, current_frame, zoom_tier=None):
        self.last_embed_frame = current_frame
        self.zoom_tier = zoom_tier
        self.names_history.append(name)
        self.similarities_history.append(similarity)
        if len(self.names_history) >= 2:
            name_scores = defaultdict(float)
            for i, (n, s) in enumerate(zip(self.names_history, self.similarities_history)):
                name_scores[n] += s * ((i + 1) / len(self.names_history))
            self.stable_name = max(name_scores.items(), key=lambda x: x[1])[0]
            self.stable_sim = float(np.mean(list(self.similarities_history)[-3:]))
        else:
            self.stable_name = name
            self.stable_sim = similarity

    def update_bbox(self, bbox, det_score, frame_num):
        self.bbox = bbox
        self.det_score = det_score
        self.bbox_history.append(bbox)
        self.last_seen_frame = frame_num
        self.frames_seen += 1

class TrackManager:
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.frame_num = 0

    def compute_iou(self, b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / (union + 1e-6)

    def update(self, detections):
        self.frame_num += 1
        results = []
        for bbox, aligned_face, det_score, landmarks, zoom_info in detections:
            best_iou, best_track = 0, None
            for tid, track in self.tracks.items():
                iou = self.compute_iou(bbox, track.bbox)
                if iou > best_iou and iou > IOU_THRESHOLD:
                    best_iou, best_track = iou, tid
            if best_track is not None:
                track = self.tracks[best_track]
                track.update_bbox(bbox, det_score, self.frame_num)
                results.append((track, aligned_face, track.needs_new_embedding(self.frame_num), zoom_info))
            else:
                track = FaceTrack(self.next_id, bbox, det_score, self.frame_num)
                self.tracks[self.next_id] = track
                self.next_id += 1
                results.append((track, aligned_face, True, zoom_info))
        for tid in [t for t, tr in self.tracks.items() if self.frame_num - tr.last_seen_frame > 20]:
            del self.tracks[tid]
        return results

track_manager = TrackManager()

# ==================== ENROLL KNOWN FACES (GPU) ====================
print(f"[INFO] Enrolling known faces from {KNOWN_FACES_DIR} ...")
gallery_names, _gal_list = [], []
for f in sorted(os.listdir(KNOWN_FACES_DIR)):
    if not f.lower().endswith(VALID_EXT):
        continue
    name = os.path.splitext(f)[0]
    img = cv2.imread(os.path.join(KNOWN_FACES_DIR, f))
    if img is None:
        print(f"  [skip] cannot read {f}"); continue
    faces = face_detector.get(img)
    if not faces:
        print(f"  [skip] no face detected in {f}"); continue
    face = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    aligned = face_align.norm_crop(img, face.kps)
    gallery_names.append(name); _gal_list.append(get_embedding(aligned))
    print(f"  [ok] enrolled {name}")

gallery_embs = np.array(_gal_list, dtype=np.float32)
if len(gallery_names) == 0:
    print("[ERROR] No known faces enrolled — check Known_faces folder."); exit()
print(f"[INFO] Enrolled {len(gallery_names)}: {', '.join(gallery_names)}\n")

# ==================== MATCHING ====================
def embed_and_match(face112):
    """112x112 BGR aligned crop -> (name, sim, runner_up_name, runner_up_sim)."""
    emb = get_embedding(face112)
    sims = gallery_embs @ emb
    order = np.argsort(sims)[::-1]
    i = int(order[0])
    j = int(order[1]) if len(order) > 1 else i
    return gallery_names[i], float(sims[i]), gallery_names[j], float(sims[j])

# ==================== VIDEO / CAMERA ====================
if VIDEO_SOURCE == 0:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"[ERROR] Could not open source: {VIDEO_SOURCE}"); exit()

frame_count = 0
prev_time = time.time()
stats = {'faces_detected': 0, 'faces_processed': 0, 'quality_rejected': 0,
         'embeddings_generated': 0, 'embeddings_cached': 0, 'zoom_tiers': defaultdict(int)}

print("[INFO] Recognition active. Press 'q' to quit, 's' for stats.\n")

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
        small = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
        all_faces = []
        for scale in (SCALES if ENABLE_MULTISCALE else [1.0]):
            scaled = small if scale == 1.0 else cv2.resize(small, None, fx=scale, fy=scale)
            try:
                bboxes, kpss = det_model.detect(scaled, max_num=0, metric='default')
            except Exception:
                continue
            if bboxes is None or len(bboxes) == 0:
                continue
            for k in range(bboxes.shape[0]):
                bbox = bboxes[k, :4].copy()
                score = float(bboxes[k, 4])
                kps = kpss[k].copy()
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
            face_size = min(w, h)
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            stats['faces_detected'] += 1
            kps_scaled = kps / DOWNSCALE
            bbox_full = (x1, y1, x2, y2)

            enhanced_region, adjusted_kps, zoom_info = zoom_extractor.extract_and_enhance(
                frame, bbox_full, kps_scaled, face_size)
            if enhanced_region is not None and zoom_info['zoomed']:
                aligned_face = face_align.norm_crop(enhanced_region, adjusted_kps)
                stats['zoom_tiers'][zoom_info['tier']] += 1
            else:
                aligned_face = face_align.norm_crop(frame, kps_scaled)
                zoom_info = {'zoomed': False, 'tier': 'none'}

            if not is_face_quality_good(aligned_face):
                stats['quality_rejected'] += 1
                continue
            face112 = aligned_face if aligned_face.shape[:2] == (112, 112) \
                      else cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            detections.append((bbox_full, face112, det_score, kps_scaled, zoom_info))

    track_results = track_manager.update(detections)

    for track, face112, needs_embed, zoom_info in track_results:
        stats['faces_processed'] += 1
        if needs_embed:
            name, sim, name2, sim2 = embed_and_match(face112)
            stats['embeddings_generated'] += 1
            track.update_embedding(name, sim, track_manager.frame_num, zoom_info.get('tier'))
            x1, y1, x2, y2 = track.bbox
            fs = min(x2 - x1, y2 - y1)
            decision = name if sim >= SIMILARITY_THRESHOLD else f"Unknown#{track.track_id}"
            print(f"[REC] f{track_manager.frame_num:<5d} trk{track.track_id:<2d} "
                  f"{fs:>3d}px {zoom_info.get('tier','none'):<9s} | "
                  f"best={name:<12s} sim={sim:.3f} | 2nd={name2:<12s} sim={sim2:.3f} | "
                  f"margin={sim - sim2:+.3f} -> {decision}")
        else:
            stats['embeddings_cached'] += 1

    for tid, track in track_manager.tracks.items():
        if track.frames_seen < 1:
            continue
        label, sim = track.stable_name, track.stable_sim
        if sim >= 0.60:
            color, conf = (0, 255, 0), "HIGH"
        elif sim >= SIMILARITY_THRESHOLD:
            color, conf = (0, 200, 255), "MED"
        else:
            label, color, conf = f"Unknown #{track.track_id}", (0, 0, 255), "?"
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({conf})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(display, text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(display, f"{min(x2-x1, y2-y1)}px | Sim:{sim:.2f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    fps = 1 / (now - prev_time + 1e-6)
    prev_time = now
    if frame_count % 30 == 0:
        print(f"[FPS] {fps:5.1f} | frame {frame_count} | active tracks: {len(track_manager.tracks)}")

    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if display.shape[1] > DISPLAY_MAX_WIDTH:
        sc = DISPLAY_MAX_WIDTH / display.shape[1]
        display = cv2.resize(display, None, fx=sc, fy=sc)
    cv2.imshow("FR2 - Far-Face Recognition (GPU)", display)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:
        break
    elif key == ord('s'):
        print("\n" + "=" * 50)
        for k, v in stats.items():
            if k != 'zoom_tiers':
                print(f"  {k}: {v}")
        print(f"  zoom_tiers: {dict(stats['zoom_tiers'])}")
        print("=" * 50 + "\n")

cap.release()
cv2.destroyAllWindows()
print("\n[INFO] Session complete")