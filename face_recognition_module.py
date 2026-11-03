"""
face_recognition_module.py — drone face recognition for GCS integration. FINAL.

Self-contained, importable, no CLI, no side effects on import.

QUICK START
-----------
    from face_recognition_module import FaceRecognizer, Config

    fr = FaceRecognizer(Config(
        known_faces_dir = r"C:\\dev\\face-recognition-pipeline\\Known_faces",
        cos_threshold   = 0.30,      # <-- CALIBRATE THIS. It is the single biggest lever.
        distance_k      = 243.0,     # optional, enables metres
        torch_lib_dir   = r"C:\\dev\\face-recognition-pipeline\\faceenv\\Lib\\site-packages\\torch\\lib",
    ))

    while streaming:
        frame = get_frame()                      # BGR numpy array
        for f in fr.process_frame(frame):
            if f.accepted:
                telemetry(f.name, f.cosine, f.bbox, f.distance_m)
        fr.draw(frame)                           # optional overlay
    fr.reset()                                   # when the stream changes


THE ONE THING THAT MATTERS: cos_threshold
-----------------------------------------
An inherited, never-validated 0.40 was costing more recall than every model-side idea we tried
combined. Sweeping it, at unchanged precision:

    DroneFace   0.40 -> 24.6% recall      0.21 -> 54.8%     (+30.3 pts)
    SCface      0.40 -> 58.0% recall     0.294 -> 74.9%     (+16.9 pts)

Calibrate it on YOUR data before deploying:

    res = fr.calibrate_threshold(genuine_dir=..., stranger_dir=..., target_precision=1.0)
    print(res); print(res.threshold)

Choose by PRECISION, not by stranger false-accepts alone. Precision counts BOTH error types:
strangers wrongly named AND one enrolled person mistaken for another. Matching only on stranger
false-accepts is what made a score-normalisation scheme look like a +32 point win when it was
actually buying recall with wrong names.


WHAT IS INSIDE (each piece earned its place)
    tiled detection        SCRFD downscales every frame to det_size, so a 40px face in 4K becomes
                           ~13px and vanishes. Tiling at native resolution recovers far faces.
    mugshot fallback       SCRFD fails when a face FILLS the frame (ID-card style reference
                           photos). Re-detecting on a padded copy fixed 100% -> 0% enrollment
                           failures on SCface's cropped mugshots.
    temporal aggregation   averages a track's embeddings; makes identity stable across frames.
    distance estimate      d = k / face_px. Pure inverse law (fitted offset 0.07 ~ 0).
                           0.95 m MAE, or 0.78 m with a per-identity k.
    quality gates          min_face_px kills junk detections (a 14px "face" that was not a face);
                           min_frames suppresses one-frame flickers.

WHAT WAS TRIED AND REJECTED (do not re-litigate without new evidence)
    backbone fine-tuning on ~13 identities  -> overfits, destroys stranger rejection
    degradation-invariant adapter           -> zero gain; degradation preserves RANKING, and
                                               top-1 held at 93.8% while cosine fell to 0.509
    adaptive score normalisation (AS-Norm)  -> loses at every precision target we care about;
                                               on DroneFace it could not reach precision 1.0 at
                                               all, while plain cosine does at 0.325

THREAD SAFETY
    One FaceRecognizer per thread. The ONNX sessions are not safe for concurrent calls.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = ["Config", "FaceResult", "CalibrationResult", "FaceRecognizer"]

_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # --- gallery: a folder of references, or a prebuilt gallery.npz ---
    known_faces_dir: Optional[str] = None   # Known_faces/<Name>.jpg  OR  Known_faces/<Name>/*.jpg
    gallery_npz: Optional[str] = None       # output of 19_build_gallery.py (takes precedence)

    # --- decision ---
    cos_threshold: float = 0.35             # CALIBRATE THIS. 0.40 is too strict; see the header.
    margin_min: float = 0.0                 # require (top1 - top2) >= this; 0 disables
    min_face_px: int = 20                   # below this the detector returns junk
    min_frames: int = 2                     # a track must persist this long to be named

    # --- detection ---
    det_size: int = 1280
    min_det_score: float = 0.30
    tile_if_wider_than: int = 2000          # tiled detection above this frame width
    tile_grid: Tuple[int, int] = (2, 2)
    tile_overlap: float = 0.20
    enhance_low_light: bool = True          # CLAHE + gamma, for DETECTION only

    # --- tracking ---
    iou_threshold: float = 0.25
    max_age_frames: int = 30

    # --- recognition ---
    denoise: bool = True                    # bilateral filter before embedding
    model_name: str = "buffalo_l"
    gpu_id: int = 0
    torch_lib_dir: Optional[str] = None     # Windows: dir holding the CUDA DLLs

    # --- distance (optional) ---
    distance_k: Optional[float] = None      # d = k / face_px
    distance_k_per_identity: Dict[str, float] = field(default_factory=dict)

    verbose: bool = True


@dataclass
class CalibrationResult:
    threshold: float
    criterion: str
    precision: float
    recall: float
    f1: float
    n_genuine: int
    n_stranger: int
    n_wrong_name: int          # accepted, but matched the WRONG enrolled person
    n_stranger_accepted: int   # a non-watchlist person given a name
    sweep: List[Tuple[float, float, float, int, int]] = field(default_factory=list)

    def __str__(self) -> str:
        out = [f"cos_threshold = {self.threshold:.4f}   ({self.criterion})",
               f"  precision {self.precision:.4f}   recall {self.recall:.1%}   F1 {self.f1:.4f}",
               f"  genuine {self.n_genuine}, strangers {self.n_stranger}",
               f"  errors: {self.n_wrong_name} wrong-name, {self.n_stranger_accepted} stranger-accepted",
               "  sweep:      thr   precision    recall  wrong  strangers"]
        for thr, p, r, wr, fa in self.sweep:
            out.append(f"          {thr:7.3f} {p:10.4f} {r:9.1%} {wr:6d} {fa:10d}")
        return "\n".join(out)


@dataclass
class FaceResult:
    """One face in the current frame."""
    track_id: int
    bbox: Tuple[int, int, int, int]        # x1, y1, x2, y2 in frame coordinates
    name: str                              # identity, or "Unknown" when not accepted
    accepted: bool
    cosine: float                          # similarity to the best gallery entry
    best_match: str                        # best gallery name regardless of acceptance
    runner_up: str
    margin: float                          # cosine(top1) - cosine(top2)
    face_px: int
    frames: int
    distance_m: Optional[float] = None


# --------------------------------------------------------------------------------------
def _prepare_dlls(torch_lib_dir: Optional[str]) -> None:
    if not torch_lib_dir or not os.path.isdir(torch_lib_dir):
        return
    os.environ["PATH"] = torch_lib_dir + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(torch_lib_dir)
        except OSError:
            pass


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / (ua + 1e-6))


class _Track:
    __slots__ = ("tid", "bbox", "vec_sum", "n", "last_seen", "px")

    def __init__(self, tid, bbox, vec, px, frame_no):
        self.tid, self.bbox, self.px, self.last_seen = tid, bbox, px, frame_no
        self.vec_sum, self.n = vec.copy(), 1

    def update(self, bbox, vec, px, frame_no):
        self.bbox, self.px, self.last_seen = bbox, px, frame_no
        self.vec_sum += vec
        self.n += 1

    @property
    def embedding(self) -> np.ndarray:
        return self.vec_sum / (np.linalg.norm(self.vec_sum) + 1e-9)


# --------------------------------------------------------------------------------------
class FaceRecognizer:
    """Stateful per-stream recognizer. Call process_frame() on consecutive frames."""

    def __init__(self, config: Config):
        self.cfg = config
        _prepare_dlls(config.torch_lib_dir)

        from insightface.app import FaceAnalysis
        from insightface.utils import face_align

        self._align = face_align
        self._app = FaceAnalysis(
            name=config.model_name,
            allowed_modules=["detection", "recognition"],
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=config.gpu_id, det_size=(config.det_size, config.det_size))
        self._det = self._app.models["detection"]
        self._rec = self._app.models["recognition"]
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        self.names: List[str] = []
        self._gallery: Optional[np.ndarray] = None
        self._load_gallery()

        self._tracks: Dict[int, _Track] = {}
        self._next_id = 0
        self._frame_no = 0
        self._last_results: List[FaceResult] = []
        self._log(f"decision: cosine >= {self.cfg.cos_threshold:.3f}"
                  f"{f', margin >= {self.cfg.margin_min:.3f}' if self.cfg.margin_min > 0 else ''}")

    # ---------------- public API ----------------
    @property
    def enrolled(self) -> List[str]:
        return list(self.names)

    def reset(self) -> None:
        """Clear tracking state. Call when the video source changes."""
        self._tracks.clear()
        self._next_id = 0
        self._frame_no = 0
        self._last_results = []

    def process_frame(self, frame: np.ndarray) -> List[FaceResult]:
        """Detect, track and identify every face in one BGR frame."""
        if self._gallery is None or not self.names:
            raise RuntimeError("no gallery enrolled")
        self._frame_no += 1

        det_in = self._enhance(frame) if self.cfg.enhance_low_light else frame
        faces = self._detect(det_in, frame.shape[1] > self.cfg.tile_if_wider_than)

        results: List[FaceResult] = []
        for box, kps, _ in faces:
            x1, y1, x2, y2 = (int(v) for v in box)
            px = min(x2 - x1, y2 - y1)
            if px < self.cfg.min_face_px:
                continue
            vec = self._embed(self._align.norm_crop(frame, kps))
            tid = self._assign_track((x1, y1, x2, y2), vec, px)
            results.append(self._identify(self._tracks[tid], (x1, y1, x2, y2), px))

        self._evict_stale()
        self._last_results = results
        return results

    def process_image(self, image: np.ndarray) -> List[FaceResult]:
        """Single still image: no tracking, no temporal aggregation."""
        saved = (self._tracks, self._next_id)
        self._tracks, self._next_id = {}, 0
        try:
            return self.process_frame(image)
        finally:
            self._tracks, self._next_id = saved

    def draw(self, frame: np.ndarray, results: Optional[List[FaceResult]] = None) -> np.ndarray:
        """Overlay boxes and labels in place."""
        for r in (results if results is not None else self._last_results):
            colour = (0, 200, 0) if r.accepted else (40, 40, 220)
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            label = f"{r.name} {r.cosine:.2f}"
            if r.distance_m is not None:
                label += f" {r.distance_m:.1f}m"
            label += f" {r.face_px}px"
            cv2.rectangle(frame, (x1, max(0, y1 - 20)), (x1 + 9 * len(label), y1), colour, -1)
            cv2.putText(frame, label, (x1 + 2, max(13, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def estimate_distance(self, face_px: float, name: Optional[str] = None) -> Optional[float]:
        """d = k / face_px. None until distance_k is configured."""
        k = self.cfg.distance_k_per_identity.get(name or "", self.cfg.distance_k)
        if not k or face_px <= 0:
            return None
        return float(k) / float(face_px)

    def calibrate_threshold(
        self,
        genuine_dir: str,
        stranger_dir: str,
        target_precision: float = 1.0,
        apply: bool = True,
        grid: Optional[Sequence[float]] = None,
    ) -> CalibrationResult:
        """Derive cos_threshold from YOUR data. Run this before deploying.

        genuine_dir   one folder per enrolled person, holding probe images OF that person:
                          genuine_dir/<Name>/*.jpg      <Name> must match an enrolled identity
                      Use real operational imagery, NOT the enrollment photos.
        stranger_dir  images of people NOT on the watchlist (searched recursively). Draw them
                      from the same cameras and conditions as deployment.
        target_precision  the highest-recall threshold achieving at least this precision is
                      chosen. 1.0 = never name anyone wrongly (costs recall). ~0.95 = more
                      recall, occasional error - appropriate when a human reviews each alert.

        Precision here counts BOTH failure modes: a stranger given a name, and one enrolled
        person mistaken for another. Selecting on stranger false-accepts alone is exactly the
        mistake that made a score-normalisation scheme look like a large win.
        """
        if not os.path.isdir(genuine_dir):
            raise RuntimeError(f"genuine_dir not found: {genuine_dir}")
        if not os.path.isdir(stranger_dir):
            raise RuntimeError(f"stranger_dir not found: {stranger_dir}")

        gen_sim, gen_ok = [], []
        for person in sorted(os.listdir(genuine_dir)):
            pdir = os.path.join(genuine_dir, person)
            if not os.path.isdir(pdir):
                continue
            if person not in self.names:
                self._log(f"WARNING: '{person}' is not enrolled - skipping its probes")
                continue
            for vec in self._embed_folder(pdir, True):
                sims = self._gallery @ vec
                i = int(np.argmax(sims))
                gen_sim.append(float(sims[i])); gen_ok.append(self.names[i] == person)
        if not gen_sim:
            raise RuntimeError("no usable genuine probes found")

        str_sim = []
        for vec in self._embed_folder(stranger_dir, True):
            str_sim.append(float(np.max(self._gallery @ vec)))
        if not str_sim:
            raise RuntimeError("no usable stranger images found")

        gen_sim = np.asarray(gen_sim); gen_ok = np.asarray(gen_ok, bool)
        str_sim = np.asarray(str_sim)
        g = np.round(np.arange(0.15, 0.61, 0.01), 3) if grid is None else np.asarray(grid, float)

        def stats(thr):
            acc = gen_sim >= thr
            tp = int(np.sum(acc & gen_ok)); wrong = int(np.sum(acc & ~gen_ok))
            fa = int(np.sum(str_sim >= thr))
            prec = tp / max(tp + wrong + fa, 1)
            rec = tp / len(gen_sim)
            return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9), wrong, fa

        sweep = [(float(t), *stats(float(t))[:2], stats(float(t))[3], stats(float(t))[4])
                 for t in g]
        eligible = [(t, p, r, wr, fa) for (t, p, r, wr, fa) in sweep if p >= target_precision]
        if eligible:
            thr = max(eligible, key=lambda e: e[2])[0]
            crit = f"highest recall with precision >= {target_precision:.4f}"
        else:
            thr = max(sweep, key=lambda e: e[1])[0]
            best_p = max(p for _, p, _, _, _ in sweep)
            crit = (f"target precision {target_precision:.4f} UNREACHABLE "
                    f"(best {best_p:.4f}) - picked max precision")
            self._log(f"WARNING: {crit}")

        p, r, f1, wr, fa = stats(thr)
        res = CalibrationResult(threshold=float(thr), criterion=crit, precision=p, recall=r, f1=f1,
                                n_genuine=len(gen_sim), n_stranger=len(str_sim),
                                n_wrong_name=wr, n_stranger_accepted=fa,
                                sweep=[s for s in sweep if s[0] % 0.05 < 1e-6 or s[0] == thr])
        if apply:
            self.cfg.cos_threshold = float(thr)
            self._log(f"applied cos_threshold = {thr:.4f}")
        self._log(str(res))
        return res

    # ---------------- setup ----------------
    def _log(self, msg: str) -> None:
        if self.cfg.verbose:
            print(f"[FaceRecognizer] {msg}")

    def _load_gallery(self) -> None:
        cfg = self.cfg
        if cfg.gallery_npz and os.path.exists(cfg.gallery_npz):
            d = np.load(cfg.gallery_npz, allow_pickle=True)
            self.names = [str(n) for n in d["names"]]
            self._gallery = d["embs"].astype(np.float32)
            self._log(f"gallery from {os.path.basename(cfg.gallery_npz)}: {len(self.names)} identities")
            return
        if not cfg.known_faces_dir or not os.path.isdir(cfg.known_faces_dir):
            raise RuntimeError("provide known_faces_dir or gallery_npz")

        names, vecs = [], []
        for entry in sorted(os.listdir(cfg.known_faces_dir)):
            full = os.path.join(cfg.known_faces_dir, entry)
            if os.path.isdir(full) and not entry.startswith("_"):
                person = entry
                files = [os.path.join(full, f) for f in sorted(os.listdir(full))
                         if f.lower().endswith(_IMG_EXT)]
            elif entry.lower().endswith(_IMG_EXT):
                person, files = os.path.splitext(entry)[0], [full]
            else:
                continue
            per = []
            for path in files:
                img = cv2.imread(path)
                if img is None:
                    continue
                crop, _ = self._align_largest(img)
                if crop is not None:
                    per.append(self._embed(crop))
            if not per:
                self._log(f"WARNING: no usable reference for '{person}' - cannot be recognised")
                continue
            m = np.mean(per, axis=0)
            names.append(person)
            vecs.append((m / (np.linalg.norm(m) + 1e-9)).astype(np.float32))
        if not names:
            raise RuntimeError(f"no usable references in {cfg.known_faces_dir}")
        self.names, self._gallery = names, np.asarray(vecs, np.float32)
        self._log(f"enrolled {len(names)}: {', '.join(names)}")

    # ---------------- pipeline ----------------
    def _enhance(self, img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        out = cv2.cvtColor(cv2.merge([self._clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
        mean = out.mean()
        if mean < 110:
            gm = 1.6 if mean < 70 else 1.3
            lut = np.array([((i / 255.0) ** (1.0 / gm)) * 255 for i in range(256)], np.uint8)
            out = cv2.LUT(out, lut)
        return out

    def _align_largest(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """Aligned 112x112 crop of the largest face, with fallbacks for tight face crops.

        SCRFD fails when a face FILLS the frame: no context, and after upscaling to det_size the
        face exceeds its anchor range. That is exactly what ID-card / mugshot reference photos
        look like - SCface's cropped mugshots failed 100% until the padded retry was added.
        """
        bb, kps = self._det.detect(img, max_num=0, metric="default")
        if bb is not None and len(bb):
            j = int(np.argmax((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))
            px = float(min(bb[j, 2] - bb[j, 0], bb[j, 3] - bb[j, 1]))
            return self._align.norm_crop(img, kps[j]), px

        h, w = img.shape[:2]
        pad = int(0.4 * max(h, w))
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        bb, kps = self._det.detect(padded, max_num=0, metric="default")
        if bb is not None and len(bb):
            j = int(np.argmax((bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])))
            px = float(min(bb[j, 2] - bb[j, 0], bb[j, 3] - bb[j, 1]))
            return self._align.norm_crop(padded, kps[j]), px

        if h and w and 0.6 <= w / h <= 1.8 and min(h, w) >= 60:
            return cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA), float(min(h, w))
        return None, 0.0

    def _detect(self, img: np.ndarray, use_tiles: bool):
        cand = []
        bb, kps = self._det.detect(img, max_num=0, metric="default")
        if bb is not None:
            for i in range(len(bb)):
                if bb[i, 4] >= self.cfg.min_det_score:
                    cand.append((bb[i, :4].copy(), kps[i].copy(), float(bb[i, 4])))
        if use_tiles:
            H, W = img.shape[:2]
            rows, cols = self.cfg.tile_grid
            th, tw = H // rows, W // cols
            oy, ox = int(th * self.cfg.tile_overlap), int(tw * self.cfg.tile_overlap)
            for r in range(rows):
                for c in range(cols):
                    y1, y2 = max(0, r * th - oy), min(H, (r + 1) * th + oy)
                    x1, x2 = max(0, c * tw - ox), min(W, (c + 1) * tw + ox)
                    tb, tk = self._det.detect(img[y1:y2, x1:x2], max_num=0, metric="default")
                    if tb is None:
                        continue
                    for i in range(len(tb)):
                        if tb[i, 4] < self.cfg.min_det_score:
                            continue
                        box = tb[i, :4].copy()
                        box[0] += x1; box[2] += x1; box[1] += y1; box[3] += y1
                        k = tk[i].copy(); k[:, 0] += x1; k[:, 1] += y1
                        cand.append((box, k, float(tb[i, 4])))
        cand.sort(key=lambda t: -t[2])
        kept = []
        for c in cand:
            if not any(_iou(c[0], k[0]) > 0.4 for k in kept):
                kept.append(c)
        return kept

    def _embed(self, face112: np.ndarray) -> np.ndarray:
        if self.cfg.denoise:
            face112 = cv2.bilateralFilter(face112, 5, 50, 50)
        f = self._rec.get_feat(face112).flatten().astype(np.float32)
        return f / (np.linalg.norm(f) + 1e-9)

    def _embed_folder(self, folder: str, recursive: bool):
        if recursive:
            paths = [os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs
                     if f.lower().endswith(_IMG_EXT)]
        else:
            paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                     if f.lower().endswith(_IMG_EXT)]
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                continue
            crop, _ = self._align_largest(img)
            if crop is not None:
                yield self._embed(crop)

    def _assign_track(self, bbox, vec, px) -> int:
        best, tid_best = 0.0, None
        for tid, tr in self._tracks.items():
            v = _iou(bbox, tr.bbox)
            if v > best and v > self.cfg.iou_threshold:
                best, tid_best = v, tid
        if tid_best is None:
            tid = self._next_id
            self._tracks[tid] = _Track(tid, bbox, vec, px, self._frame_no)
            self._next_id += 1
            return tid
        self._tracks[tid_best].update(bbox, vec, px, self._frame_no)
        return tid_best

    def _evict_stale(self) -> None:
        for t in [t for t, tr in self._tracks.items()
                  if self._frame_no - tr.last_seen > self.cfg.max_age_frames]:
            del self._tracks[t]

    def _identify(self, track: _Track, bbox, px) -> FaceResult:
        sims = self._gallery @ track.embedding
        order = np.argsort(sims)[::-1]
        i = int(order[0])
        j = int(order[1]) if len(order) > 1 else i
        cos1, cos2 = float(sims[i]), float(sims[j])
        margin = cos1 - cos2

        accepted = cos1 >= self.cfg.cos_threshold
        if self.cfg.margin_min > 0 and margin < self.cfg.margin_min:
            accepted = False
        if track.n < self.cfg.min_frames:
            accepted = False

        name = self.names[i]
        return FaceResult(
            track_id=track.tid, bbox=tuple(bbox),
            name=name if accepted else "Unknown", accepted=accepted,
            cosine=cos1, best_match=name, runner_up=self.names[j], margin=margin,
            face_px=int(px), frames=track.n,
            distance_m=self.estimate_distance(px, name) if accepted else None,
        )


# --------------------------------------------------------------------------------------
# OPERATIONAL NOTES
# --------------------------------------------------------------------------------------
# FACE SIZE governs everything. Measured recall by face pixels:
#     <35px 2%   35-45px 5%   45-55px 16%   55-70px 28%   70-90px 49%   90-120px 63%
#     50% recall at ~66px, 80% at ~102px. Below ~45px the identity signal is largely absent -
#     that is optics and flight altitude, not software. Optical zoom moves the whole curve right.
#
# CAMERA PITCH is a second, independent factor. Recall by drone height: 1.5m 33%, 3m 33%,
#     4m 21%, 5m 11% - and at IDENTICAL face pixel counts recall still went 0.875 -> 0.000.
#     Fly low with standoff so the view is near-horizontal; do not hover high overhead.
#
# DARKNESS barely affects recognition (-0.08 cosine even at 7% brightness). Low light breaks
#     DETECTION, not identity. NOISE is the real killer - it breaks identity at sigma=10.
#
# PERFORMANCE: tiled detection costs ~5 detector passes per frame. For real-time, call
#     process_frame() every Nth frame; tracking and temporal aggregation keep identity stable.
#
# CALIBRATING distance_k: put a person at a known distance D, read face_px, set k = D * face_px.
#     A per-identity k cut error 0.95 m -> 0.78 m (real face widths vary ~21% between people).
#
# MEASURED BEHAVIOUR (expectation-setting, not guarantees):
#     Real drone video, single-photo enrollment   precision 0.950, recall 0.753
#     SCface 1.0 m / 2.6 m / 4.2 m                97% / 83% / 20%
#     Threshold tuning alone                      +17 to +30 pts recall at unchanged precision
