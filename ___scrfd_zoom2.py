"""
SCRFD WITH AGGRESSIVE FAR-FACE ZOOM
Optimized for faces at 3-5+ meters distance
Extracts maximum context and uses multi-stage upscaling
"""
import os
os.add_dll_directory(r"C:\dev\face-recognition-pipeline\faceenv\Lib\site-packages\torch\lib")
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from deepface import DeepFace
import pickle
import time
from collections import deque, defaultdict

# ==================== CONFIG ====================
MODEL_NAME = "ArcFace"
DETECTOR_NAME = "buffalo_l"

EMBEDDING_CACHE_FILE = "embeddings_scrfd.pkl"
KNOWN_FACES_DIR = "known_faces"
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# Detection settings
FRAME_SKIP = 2  # More frequent detection for far faces
DOWNSCALE = 0.7  # Less downscaling = better far face detection
MIN_FACE_SIZE = 12
MIN_DETECTION_SCORE = 0.25  # Lower threshold for far faces

# Multi-scale detection (CRITICAL for far faces)
ENABLE_MULTISCALE = True
SCALES = [1.0, 1.5, 2.0, 2.5, 3.0]  # Added 3.0x for very far faces

# AGGRESSIVE FAR-FACE ZOOM SETTINGS
ENABLE_FAR_FACE_ZOOM = True

# Multi-tier zoom strategy
ZOOM_TIERS = {
    'very_far': {
        'threshold': 25,           # Faces < 25px (4-5m+ distance)
        'context_multiplier': 4.0,  # Extract 4x larger region
        'target_size': 240,         # Upscale to 240px
        'denoise': True,
        'sharpen': True
    },
    'far': {
        'threshold': 40,           # Faces 25-40px (3-4m distance)
        'context_multiplier': 3.5,  # Extract 3.5x larger region
        'target_size': 200,         # Upscale to 200px
        'denoise': True,
        'sharpen': True
    },
    'medium_far': {
        'threshold': 60,           # Faces 40-60px (2-3m distance)
        'context_multiplier': 2.5,  # Extract 2.5x larger region
        'target_size': 160,         # Upscale to 160px
        'denoise': False,
        'sharpen': True
    }
}

# Recognition thresholds (lowered for far faces)
SIMILARITY_THRESHOLD = 0.40  # Max-precision operating point from 03_validate.py (P=0.97)

# Track-based caching
EMBED_REFRESH_FRAMES = 8  # More frequent refresh for far faces
IOU_THRESHOLD = 0.3

# Quality filtering (more lenient for far faces)
ENABLE_QUALITY_FILTER = True
MIN_BLUR_THRESHOLD = 50  # Lowered from 100

# ==================== SETUP ====================
os.environ['TMPDIR'] = os.path.join(os.getcwd(), "tmp")
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs("tmp", exist_ok=True)

print("=" * 70)
print("AGGRESSIVE FAR-FACE ZOOM - OPTIMIZED FOR 3-5M+ DISTANCE")
print("=" * 70)
print("\nZoom Tiers:")
for tier_name, tier in ZOOM_TIERS.items():
    print(f"  {tier_name:12s}: <{tier['threshold']}px → {tier['context_multiplier']}x context, {tier['target_size']}px target")
print()

# ==================== SCRFD DETECTOR ====================
print("[INFO] Initializing SCRFD detector...")
face_detector = FaceAnalysis(
    name=DETECTOR_NAME,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
face_detector.prepare(ctx_id=0, det_size=(960, 960))
print("[INFO] SCRFD ready\n")

# ==================== AGGRESSIVE ZOOM EXTRACTOR ====================
class AggressiveFarFaceZoom:
    """
    Multi-tier zoom system for very distant faces
    
    Strategy:
    - Very far faces (<25px): 4x context, heavy enhancement
    - Far faces (25-40px): 3.5x context, moderate enhancement
    - Medium-far (40-60px): 2.5x context, light enhancement
    - Close faces (>60px): No zoom, normal processing
    """
    
    def __init__(self, zoom_tiers):
        self.zoom_tiers = zoom_tiers
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    
    def get_zoom_tier(self, face_size):
        """Determine which zoom tier to use based on face size"""
        if face_size < self.zoom_tiers['very_far']['threshold']:
            return 'very_far', self.zoom_tiers['very_far']
        elif face_size < self.zoom_tiers['far']['threshold']:
            return 'far', self.zoom_tiers['far']
        elif face_size < self.zoom_tiers['medium_far']['threshold']:
            return 'medium_far', self.zoom_tiers['medium_far']
        else:
            return None, None
    
    def denoise_strong(self, img):
        """Strong denoising for very far faces"""
        return cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    
    def sharpen_adaptive(self, img, intensity='medium'):
        """Adaptive sharpening based on face distance"""
        if intensity == 'strong':
            kernel = np.array([[-1,-1,-1], [-1, 10,-1], [-1,-1,-1]]) / 2
        elif intensity == 'medium':
            kernel = np.array([[0,-1,0], [-1, 6,-1], [0,-1,0]]) / 2
        else:  # light
            kernel = np.array([[0,-1,0], [-1, 5,-1], [0,-1,0]]) / 1
        
        return cv2.filter2D(img, -1, kernel)
    
    def enhance_contrast(self, img):
        """CLAHE contrast enhancement"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    
    def extract_and_enhance(self, frame, bbox, landmarks, face_size):
        """
        Extract context region and apply tier-appropriate enhancement
        
        Returns:
            enhanced_region: Processed region ready for alignment
            adjusted_landmarks: Landmarks adjusted to new coordinate system
            zoom_info: Metadata about zoom operation
        """
        tier_name, tier_config = self.get_zoom_tier(face_size)
        
        if tier_config is None:
            # Face is close enough - no zoom needed
            return None, landmarks, {
                'zoomed': False, 
                'tier': 'none',
                'face_size': face_size
            }
        
        x1, y1, x2, y2 = bbox
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Calculate expanded region
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        multiplier = tier_config['context_multiplier']
        expanded_w = face_w * multiplier
        expanded_h = face_h * multiplier
        
        # Calculate bounds (with frame boundary checks)
        new_x1 = int(max(0, center_x - expanded_w / 2))
        new_y1 = int(max(0, center_y - expanded_h / 2))
        new_x2 = int(min(frame.shape[1], center_x + expanded_w / 2))
        new_y2 = int(min(frame.shape[0], center_y + expanded_h / 2))
        
        # Extract region
        extracted = frame[new_y1:new_y2, new_x1:new_x2].copy()
        
        if extracted.size == 0:
            return None, landmarks, {'zoomed': False, 'tier': 'failed', 'face_size': face_size}
        
        # Multi-stage enhancement pipeline
        enhanced = extracted.copy()
        
        # Stage 1: Denoising (for very noisy far faces)
        if tier_config['denoise']:
            if tier_name == 'very_far':
                enhanced = self.denoise_strong(enhanced)
            else:
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
        
        # Stage 2: Upscaling to target size
        actual_w = new_x2 - new_x1
        actual_h = new_y2 - new_y1
        
        if actual_w > 0 and actual_h > 0:
            target = tier_config['target_size']
            scale_factor = target / max(actual_w, actual_h)
            new_w = int(actual_w * scale_factor)
            new_h = int(actual_h * scale_factor)
            
            # Use high-quality interpolation
            enhanced = cv2.resize(enhanced, (new_w, new_h), 
                                 interpolation=cv2.INTER_LANCZOS4)
        else:
            scale_factor = 1.0
        
        # Stage 3: Contrast enhancement
        enhanced = self.enhance_contrast(enhanced)
        
        # Stage 4: Adaptive sharpening
        if tier_config['sharpen']:
            if tier_name == 'very_far':
                enhanced = self.sharpen_adaptive(enhanced, 'strong')
            elif tier_name == 'far':
                enhanced = self.sharpen_adaptive(enhanced, 'medium')
            else:
                enhanced = self.sharpen_adaptive(enhanced, 'light')
        
        # Adjust landmarks to new coordinate system
        adjusted_landmarks = landmarks.copy()
        adjusted_landmarks[:, 0] = (adjusted_landmarks[:, 0] - new_x1) * scale_factor
        adjusted_landmarks[:, 1] = (adjusted_landmarks[:, 1] - new_y1) * scale_factor
        
        zoom_info = {
            'zoomed': True,
            'tier': tier_name,
            'face_size': face_size,
            'original_bbox': bbox,
            'zoom_bbox': (new_x1, new_y1, new_x2, new_y2),
            'context_multiplier': multiplier,
            'scale_factor': scale_factor,
            'target_size': tier_config['target_size'],
            'extracted_size': (enhanced.shape[1], enhanced.shape[0])
        }
        
        return enhanced, adjusted_landmarks, zoom_info

zoom_extractor = AggressiveFarFaceZoom(ZOOM_TIERS)

# ==================== PREPROCESSOR ====================
class RobustPreprocessor:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    
    def adaptive_gamma(self, img):
        mean_luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
        
        if mean_luma < 80:
            gamma = 1.7
        elif mean_luma < 130:
            gamma = 1.3
        elif mean_luma < 170:
            gamma = 1.1
        else:
            gamma = 1.0
        
        if gamma == 1.0:
            return img
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        return cv2.LUT(img, table)
    
    def enhance(self, img):
        gamma_corrected = self.adaptive_gamma(img)
        
        lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    def process(self, img):
        enhanced = self.enhance(img)
        
        if enhanced.shape[0] != 112 or enhanced.shape[1] != 112:
            enhanced = cv2.resize(enhanced, (112, 112), interpolation=cv2.INTER_LANCZOS4)
        
        return enhanced

preprocessor = RobustPreprocessor()

# ==================== QUALITY FILTER ====================
def is_face_quality_good(face_img):
    if not ENABLE_QUALITY_FILTER:
        return True
    
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return blur_score >= MIN_BLUR_THRESHOLD

# ==================== FACE TRACK ====================
class FaceTrack:
    def __init__(self, track_id, bbox, det_score, frame_num):
        self.track_id = track_id
        self.bbox = bbox
        self.det_score = det_score
        self.last_seen_frame = frame_num
        
        self.cached_embedding = None
        self.last_embed_frame = -999
        
        self.embeddings_history = deque(maxlen=10)  # Increased for far faces
        self.names_history = deque(maxlen=10)
        self.similarities_history = deque(maxlen=10)
        
        self.stable_name = "Unknown"
        self.stable_sim = 0.0
        self.frames_seen = 0
        
        # Zoom tracking
        self.zoom_tier = None
        self.zoom_count = 0
        
        self.bbox_history = deque(maxlen=5)
        self.bbox_history.append(bbox)
    
    def bbox_changed_significantly(self):
        if len(self.bbox_history) < 2:
            return False
        
        current = self.bbox_history[-1]
        prev = self.bbox_history[-2]
        
        current_area = (current[2] - current[0]) * (current[3] - current[1])
        prev_area = (prev[2] - prev[0]) * (prev[3] - prev[1])
        
        area_change = abs(current_area - prev_area) / (prev_area + 1e-6)
        return area_change > 0.15
    
    def needs_new_embedding(self, current_frame):
        frames_since_embed = current_frame - self.last_embed_frame
        
        if frames_since_embed >= EMBED_REFRESH_FRAMES:
            return True
        
        if self.bbox_changed_significantly():
            return True
        
        return False
    
    def update_embedding(self, embedding, name, similarity, current_frame, zoom_tier=None):
        self.cached_embedding = embedding
        self.last_embed_frame = current_frame
        self.zoom_tier = zoom_tier
        
        if zoom_tier:
            self.zoom_count += 1
        
        self.embeddings_history.append(embedding)
        self.names_history.append(name)
        self.similarities_history.append(similarity)
        
        if len(self.names_history) >= 2:
            name_scores = defaultdict(float)
            for i, (n, s) in enumerate(zip(self.names_history, self.similarities_history)):
                weight = (i + 1) / len(self.names_history)
                name_scores[n] += s * weight
            
            self.stable_name = max(name_scores.items(), key=lambda x: x[1])[0]
            self.stable_sim = np.mean(list(self.similarities_history)[-3:])
    
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
    
    def compute_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        inter = (xi2 - xi1) * (yi2 - yi1)
        union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
        
        return inter / (union + 1e-6)
    
    def update(self, detections):
        self.frame_num += 1
        
        matched_tracks = set()
        results = []
        
        for bbox, aligned_face, det_score, landmarks, zoom_info in detections:
            best_iou = 0
            best_track = None
            
            for tid, track in self.tracks.items():
                iou = self.compute_iou(bbox, track.bbox)
                if iou > best_iou and iou > IOU_THRESHOLD:
                    best_iou = iou
                    best_track = tid
            
            if best_track is not None:
                track = self.tracks[best_track]
                track.update_bbox(bbox, det_score, self.frame_num)
                matched_tracks.add(best_track)
                
                needs_embed = track.needs_new_embedding(self.frame_num)
                results.append((track, aligned_face, needs_embed, zoom_info))
            else:
                track = FaceTrack(self.next_id, bbox, det_score, self.frame_num)
                self.tracks[self.next_id] = track
                self.next_id += 1
                
                results.append((track, aligned_face, True, zoom_info))
        
        to_remove = [
            tid for tid, track in self.tracks.items()
            if self.frame_num - track.last_seen_frame > 20
        ]
        for tid in to_remove:
            del self.tracks[tid]
        
        return results

track_manager = TrackManager()

# ==================== LOAD FINE-TUNED ADAPTER + GALLERY ====================
# Replaces the old raw-ArcFace known_faces cache. We now run the full validated
# chain:  aligned crop -> ArcFace (512) -> trained adapter (256) -> gallery match
# (PR-AUC 0.94 in 03_validate.py), instead of matching raw embeddings.
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

MODEL_DIR = r"C:\dev\train_val\models"
DEVICE    = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(os.path.join(MODEL_DIR, "class_map.json")) as fh:
    _meta = json.load(fh)
EMBED_DIM = _meta['embed_dim']
IN_DIM    = _meta['in_dim']

class MetricAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )
    def forward(self, x):
        return F.normalize(self.proj(x), dim=1)

adapter = MetricAdapter(IN_DIM, EMBED_DIM).to(DEVICE)
adapter.load_state_dict(torch.load(os.path.join(MODEL_DIR, "adapter_best.pt"), map_location=DEVICE))
adapter.eval()

# Enroll the Known_faces people through the adapter (one reference per photo).
KNOWN_FACES_DIR = r"C:\dev\face-recognition-pipeline\Known_faces"
print(f"[INFO] Enrolling known faces from {KNOWN_FACES_DIR} …")

gallery_names, _gal_list = [], []
for f in sorted(os.listdir(KNOWN_FACES_DIR)):
    if not f.lower().endswith(VALID_EXT):
        continue
    name = os.path.splitext(f)[0]
    img  = cv2.imread(os.path.join(KNOWN_FACES_DIR, f))
    if img is None:
        print(f"  [skip] cannot read {f}"); continue
    faces = face_detector.get(img)
    if not faces:
        print(f"  [skip] no face detected in {f}"); continue
    face    = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]))
    aligned = face_align.norm_crop(img, face.kps)          # 112x112
    raw = DeepFace.represent(img_path=aligned, model_name=MODEL_NAME,
                             detector_backend="skip", enforce_detection=False)[0]["embedding"]
    raw = np.array(raw, dtype=np.float32); raw /= (np.linalg.norm(raw) + 1e-9)
    with torch.no_grad():
        proj = adapter(torch.from_numpy(raw[None]).float().to(DEVICE)).cpu().numpy()[0]
    gallery_names.append(name); _gal_list.append(proj)
    print(f"  [ok] enrolled {name}")

gallery_embs = np.array(_gal_list, dtype=np.float32)        # (5, EMBED_DIM)
if len(gallery_names) == 0:
    print("[ERROR] No known faces enrolled — check Known_faces folder."); exit()
print(f"[INFO] Enrolled {len(gallery_names)}: {', '.join(gallery_names)}")

# ==================== MATCHING ====================
def embed_and_match(face112):
    """112x112 BGR aligned crop -> (name, similarity, projected_embedding)."""
    raw = DeepFace.represent(
        img_path=face112,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False
    )[0]["embedding"]
    raw = np.array(raw, dtype=np.float32)
    raw /= (np.linalg.norm(raw) + 1e-9)
    with torch.no_grad():
        proj = adapter(torch.from_numpy(raw[None]).float().to(DEVICE)).cpu().numpy()[0]
    sims = gallery_embs @ proj
    order = np.argsort(sims)[::-1]
    i = int(order[0])
    j = int(order[1]) if len(order) > 1 else i
    return gallery_names[i], float(sims[i]), proj, gallery_names[j], float(sims[j])

# ==================== CAMERA ====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
prev_time = time.time()

stats = {
    'embeddings_generated': 0,
    'embeddings_cached': 0,
    'faces_processed': 0,
    'faces_detected': 0,
    'quality_rejected': 0,
    'zoom_tiers': defaultdict(int),
    'far_faces': {'<25px': 0, '25-40px': 0, '40-60px': 0, '>60px': 0}
}

print("\n[INFO] Aggressive far-face zoom system active")
print("[INFO] Press 'q' to quit, 's' for stats, 'z' to toggle zoom preview\n")

show_zoom_preview = False
zoom_preview_images = []

# ==================== MAIN LOOP ====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    display = frame.copy()
    now = time.time()

    detections = []
    zoom_preview_images = []
    
    if frame_count % FRAME_SKIP == 0:
        small = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)
        
        all_faces = []
        
        for scale in (SCALES if ENABLE_MULTISCALE else [1.0]):
            if scale != 1.0:
                scaled = cv2.resize(small, None, fx=scale, fy=scale)
            else:
                scaled = small
            
            try:
                faces = face_detector.get(scaled)
                
                for face in faces:
                    bbox = face.bbox.copy()
                    kps = face.kps.copy()
                    
                    if scale != 1.0:
                        bbox = bbox / scale
                        kps = kps / scale
                    
                    all_faces.append({
                        'bbox': bbox,
                        'kps': kps,
                        'det_score': face.det_score
                    })
            except:
                pass
        
        # NMS
        if len(all_faces) > 1:
            all_faces = sorted(all_faces, key=lambda x: x['det_score'], reverse=True)
            
            filtered_faces = []
            for face in all_faces:
                is_duplicate = False
                for kept in filtered_faces:
                    iou = track_manager.compute_iou(face['bbox'], kept['bbox'])
                    if iou > 0.5:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    filtered_faces.append(face)
            
            all_faces = filtered_faces
        
        # Process with aggressive zoom
        for face in all_faces:
            bbox = face['bbox']
            kps = face['kps']
            det_score = face['det_score']
            
            if det_score < MIN_DETECTION_SCORE:
                continue
            
            x1 = int(bbox[0] / DOWNSCALE)
            y1 = int(bbox[1] / DOWNSCALE)
            x2 = int(bbox[2] / DOWNSCALE)
            y2 = int(bbox[3] / DOWNSCALE)
            
            w, h = x2 - x1, y2 - y1
            face_size = min(w, h)
            
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            
            stats['faces_detected'] += 1
            
            # Track face size distribution
            if face_size < 25:
                stats['far_faces']['<25px'] += 1
            elif face_size < 40:
                stats['far_faces']['25-40px'] += 1
            elif face_size < 60:
                stats['far_faces']['40-60px'] += 1
            else:
                stats['far_faces']['>60px'] += 1
            
            kps_scaled = kps / DOWNSCALE
            bbox_full = (x1, y1, x2, y2)
            
            # AGGRESSIVE ZOOM EXTRACTION
            enhanced_region, adjusted_kps, zoom_info = zoom_extractor.extract_and_enhance(
                frame, bbox_full, kps_scaled, face_size
            )
            
            if enhanced_region is not None and zoom_info['zoomed']:
                # Align from enhanced region
                aligned_face = face_align.norm_crop(enhanced_region, adjusted_kps)
                stats['zoom_tiers'][zoom_info['tier']] += 1
                
                # Store for preview
                if show_zoom_preview and len(zoom_preview_images) < 3:
                    zoom_preview_images.append({
                        'image': enhanced_region.copy(),
                        'bbox': bbox_full,
                        'tier': zoom_info['tier'],
                        'size': face_size
                    })
            else:
                # Normal alignment
                aligned_face = face_align.norm_crop(frame, kps_scaled)
                zoom_info = {'zoomed': False, 'tier': 'none'}
            
            if not is_face_quality_good(aligned_face):
                stats['quality_rejected'] += 1
                continue
            
            # Embed the plain aligned crop (no CLAHE/gamma) so live preprocessing
            # matches how the gallery was built in 00/01.
            face112 = aligned_face if aligned_face.shape[:2] == (112, 112) \
                      else cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_LANCZOS4)
            detections.append((bbox_full, face112, det_score, kps_scaled, zoom_info))

    # Update tracks
    track_results = track_manager.update(detections)
    
    # Embed
    for track, preprocessed_face, needs_embed, zoom_info in track_results:
        stats['faces_processed'] += 1
        
        if needs_embed:
            name, sim, proj, name2, sim2 = embed_and_match(preprocessed_face)
            stats['embeddings_generated'] += 1
            track.update_embedding(proj, name, sim, track_manager.frame_num,
                                 zoom_info.get('tier'))
            x1, y1, x2, y2 = track.bbox
            fs = min(x2 - x1, y2 - y1)
            decision = name if sim >= SIMILARITY_THRESHOLD else "UNKNOWN"
            print(f"[REC] f{track_manager.frame_num:<5d} trk{track.track_id:<2d} "
                  f"{fs:>3d}px {zoom_info.get('tier','none'):<9s} | "
                  f"best={name:<12s} sim={sim:.3f} | 2nd={name2:<12s} sim={sim2:.3f} | "
                  f"margin={sim - sim2:+.3f} -> {decision}")
        else:
            stats['embeddings_cached'] += 1

    # Draw
    for tid, track in track_manager.tracks.items():
        if track.frames_seen < 2:
            continue
        
        label = track.stable_name
        sim = track.stable_sim
        
        # Color by similarity — UNKNOWN below the accept threshold (precision-first)
        if sim >= 0.88:
            color = (0, 255, 0); conf = "HIGH"
        elif sim >= SIMILARITY_THRESHOLD:
            color = (0, 200, 255); conf = "MED"
        else:
            label = "UNKNOWN"; color = (0, 0, 255); conf = "?"
        
        x1, y1, x2, y2 = track.bbox
        face_size = min(x2 - x1, y2 - y1)
        
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
        
        # Zoom tier indicator
        tier_icon = ""
        if track.zoom_tier == 'very_far':
            tier_icon = " 🔍🔍"  # Double zoom
        elif track.zoom_tier == 'far':
            tier_icon = " 🔍+"
        elif track.zoom_tier == 'medium_far':
            tier_icon = " 🔍"
        
        label_text = f"{label} ({conf}){tier_icon}"
        
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(display, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        
        cv2.putText(display, label_text, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        
        info_text = f"{face_size}px | Sim:{sim:.2f}"
        cv2.putText(display, info_text,
                   (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Show zoom previews
    if show_zoom_preview and zoom_preview_images:
        y_offset = 10
        for zp in zoom_preview_images:
            img = zp['image']
            if img.shape[0] > 0 and img.shape[1] > 0:
                # Resize for preview
                preview_h = 150
                aspect = img.shape[1] / img.shape[0]
                preview_w = int(preview_h * aspect)
                preview = cv2.resize(img, (preview_w, preview_h))
                
                # Place in top-right
                x_offset = display.shape[1] - preview_w - 10
                
                display[y_offset:y_offset+preview_h, x_offset:x_offset+preview_w] = preview
                
                # Label with tier and size
                tier_label = f"{zp['tier'].upper()} ({zp['size']}px)"
                cv2.rectangle(display, (x_offset, y_offset - 25), (x_offset + preview_w, y_offset),
                            (0, 0, 0), -1)
                cv2.putText(display, tier_label, (x_offset + 5, y_offset - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                
                y_offset += preview_h + 30

    # HUD
    fps = 1 / (now - prev_time + 1e-6)
    prev_time = now

    if frame_count % 30 == 0:
        print(f"[FPS] {fps:5.1f} | frame {frame_count} | active tracks: {len(track_manager.tracks)}")

    cv2.putText(display, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.putText(display, "AGGRESSIVE ZOOM", (10, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Show zoom tier usage
    total_zoom = sum(stats['zoom_tiers'].values())
    if total_zoom > 0:
        cv2.putText(display, f"Zoomed: {total_zoom} faces", (10, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Aggressive Far-Face Zoom", display)

    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), 27]:
        break
    elif key == ord('s'):
        print("\n" + "=" * 70)
        print("AGGRESSIVE ZOOM STATISTICS")
        print("=" * 70)
        print(f"Faces detected: {stats['faces_detected']}")
        print(f"Quality rejected: {stats['quality_rejected']}")
        print(f"Faces processed: {stats['faces_processed']}")
        print(f"Embeddings generated: {stats['embeddings_generated']}")
        print(f"Embeddings cached: {stats['embeddings_cached']}")
        print("\nFace Size Distribution:")
        for size_range, count in stats['far_faces'].items():
            print(f"  {size_range:10s}: {count}")
        print("\nZoom Tier Usage:")
        for tier, count in stats['zoom_tiers'].items():
            print(f"  {tier:12s}: {count}")
        total_zoom = sum(stats['zoom_tiers'].values())
        if stats['faces_detected'] > 0:
            zoom_rate = total_zoom / stats['faces_detected'] * 100
            print(f"\nOverall zoom rate: {zoom_rate:.1f}%")
        if stats['faces_processed'] > 0:
            speedup = stats['faces_processed'] / max(stats['embeddings_generated'], 1)
            print(f"Processing speedup: {speedup:.1f}x")
        print("=" * 70 + "\n")
    elif key == ord('z'):
        show_zoom_preview = not show_zoom_preview
        print(f"Zoom preview: {'ON' if show_zoom_preview else 'OFF'}")

cap.release()
cv2.destroyAllWindows()

print("\n[INFO] Session complete")
print(f"[INFO] Total zoom operations: {sum(stats['zoom_tiers'].values())}")