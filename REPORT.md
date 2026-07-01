# Far-Face Recognition — Training & Validation Report

**Project:** Far-face recognition (3–5 m+ distance, drone footage), precision-focused
**Date:** 2026-06-25
**Repository:** face-recognition-pipeline
**Hardware:** Windows 11, RTX 5090 Laptop GPU, Python 3.11.9

---

## 1. Objective

Build and validate a face-recognition system that maximizes **precision** (avoid false
matches) on **far/low-resolution faces** typical of drone footage at 3–5 m and beyond.

The deliverable is a measured pipeline: raw photos → trained model → quantified
precision/recall, plus a real-time webcam/drone inference script.

---

## 2. Approach (summary)

We use **transfer learning + metric learning**, not training from scratch:

1. A pre-trained **ArcFace** backbone (via DeepFace) produces a 512-dim embedding per face.
2. The backbone is **frozen**.
3. A lightweight **adapter** (512→256) is trained on top with **ArcFace (angular-margin) loss**
   to specialize the embedding space to our identities and far-face conditions.
4. A per-identity **gallery** of reference embeddings is built from the training set.
5. Test faces are matched against the gallery by cosine similarity; a **threshold sweep**
   determines the precision/recall trade-off.

---

## 3. Pipeline

| Script | Role | Output |
|---|---|---|
| `00_prepare_data.py` | Detect (SCRFD) + align faces to 112×112; train/test split | `aligned/`, `manifest.json` |
| `01_finetune.py` | Extract ArcFace embeddings; train the adapter (ArcFace loss) | `adapter_best.pt`, `class_map.json` |
| `02_build_gallery.py` | Build per-identity reference vectors (top-80% by centroid) | `gallery.npz` |
| `03_validate.py` | Embed test set, threshold sweep, metrics, plots | `validation_report.txt`, `pr_curve.png`, `confusion_matrix.png`, `best_thresholds.json` |
| `___scrfd_zoom2.py` / `fr0.py` | Live inference (webcam/drone) with far-face zoom | — |

---

## 4. Dataset

Two sources, combined into **14 identities**.

| Source | Content | Identities | Notes |
|---|---|---|---|
| DnHFaces | Full photos (`photos_all`) + per-person crops (`photos_all_faces`) | 11 (a–k) | 124 crops/person; far-face thumbnails 29–216 px |
| Data2 | DJI drone far-face crops, pre-cropped 224×224 | 3 (Eni, Harshana, Abhinav) | Already cropped; no detection possible |

**Split totals after preparation:** 689 train images, 1203 test images.

### 4.1 Critical data-source correction
The original `00_prepare_data.py` read `photos_all` (multi-person **group shots** named
`ab`, `cd`, …) and kept only the largest face — collapsing **11 real people into 5 mislabeled
"identities"** (e.g. identity `ab` blended two different people). This would invalidate every
downstream metric.

**Fix:** switched the DnH source to `photos_all_faces`, which is labeled **per individual**
(a–k = 11 people). For tight crops where SCRFD cannot detect a face, we fall back to a direct
resize. Data2's pre-cropped 224×224 faces are resized to 112×112 without detection (a detector
cannot find a face that fills the entire frame).

**Result:** 0 DnH faces lost, Data2 went from 0 usable to 528 usable, and the 14 identities
are now correctly per-person.

### 4.2 Per-identity split counts

| Identity | Train | Test |
|---|---|---|
| D2_Eni | 150 | 65 |
| D2_Harshana | 187 | 115 |
| D2_Abhinav | 11 | 0 |
| DnH_a … DnH_k (×11) | 31 each | 93 each |

---

## 5. Methodology (detail)

### 5.1 Detection & alignment (`00`)
- Detector: insightface SCRFD (`buffalo_sc`), `det_size=(640, 640)`.
- Each detected face aligned to **112×112** via 5-point landmark `norm_crop` (standard ArcFace input).
- Distance-driven detection difficulty: full DnH photos are ~10 MP; default `det_size` 640 missed
  ~76% of far faces. Using the pre-extracted `photos_all_faces` crops (the chosen source) avoids
  this for training. For full-frame inference, a larger `det_size` / the live multi-scale path is used.

### 5.2 Adapter training (`01`)
- Backbone: DeepFace **ArcFace**, frozen, 512-dim embeddings (extracted on CPU via TensorFlow).
- Adapter: `Linear(512 → 256, bias=False)` + `BatchNorm1d`, output **L2-normalized**.
- Loss: **ArcFace** with scale `s = 32.0`, margin `m = 0.50`.
- Optimizer: AdamW, lr `3e-4`, weight decay `1e-4`, cosine LR schedule.
- Epochs: 40, batch size 32. Light embedding-space noise augmentation (σ=0.02).
- Device: CUDA (RTX 5090) for the adapter; embedding extraction on CPU (TensorFlow has no
  native-Windows GPU support).
- **Observed training loss:** 13.94 (epoch 1) → 2.82 (best) — steep early drop then plateau.

### 5.3 Gallery construction (`02`)
- Project each identity's train embeddings through the adapter.
- Compute the per-identity centroid; keep the **top 80%** closest to the centroid (cosine),
  discarding the lowest-quality ~20%.
- Average the retained embeddings → one L2-normalized reference vector per identity.

### 5.4 Validation (`03`)
- Embed all 1203 test images (held out; never used in training), project through the adapter.
- For each test face: cosine similarity to all 14 gallery vectors; top-1 is the prediction.
- **Decision rule:** accept if `similarity ≥ threshold`, else **abstain**.
- Counting: TP = accepted & correct; FP = accepted & wrong; FN = abstained.
  - Precision = TP / (TP + FP); Recall = TP / (TP + FN); F1 = harmonic mean.
- **Threshold sweep:** 0.10 → 0.99 in 0.01 steps; metrics recorded at each.
- Reported precision-maximizing threshold = highest precision subject to recall ≥ 0.30.
- PR-AUC = area under the precision-recall curve (computed for both adapted and raw models).

---

## 6. Results

### 6.1 Headline: adapted vs raw ArcFace (same test set)

| Metric | Raw ArcFace | Adapted (fine-tuned) |
|---|---|---|
| PR-AUC | 0.448 | **0.943** |
| Best-precision threshold | 0.10 | 0.80 |
| Precision @ best | 0.555 | **0.974** |

Raw ArcFace is near-useless on these far faces (PR-AUC 0.45). The adapter lifts it to 0.94 —
a large, measured improvement on identical held-out data.

### 6.2 Threshold trade-off (adapted model)

| Threshold | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| 0.10 | 0.848 | 1.000 | 0.918 | 1020 | 183 | 0 |
| 0.50 | 0.874 | 0.930 | 0.901 | 987 | 142 | 74 |
| 0.60 | 0.910 | 0.800 | 0.852 | 892 | 88 | 223 |
| 0.65 | 0.920 | 0.747 | 0.825 | 844 | 73 | 286 |
| **0.70** | **0.934** | **0.643** | **0.762** | 740 | 52 | 411 |
| 0.75 | 0.951 | 0.533 | 0.683 | 624 | 32 | 547 |
| 0.80 | 0.974 | 0.384 | 0.551 | 457 | 12 | 734 |
| 0.90 | 0.986 | 0.058 | 0.110 | 70 | 1 | 1132 |

**Selected operating point: 0.70** — 93.4% precision, 64.3% recall (precision-leaning balance).

### 6.3 Per-identity (at threshold 0.80)

| Identity | N | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| D2_Eni | 65 | 60 | 0 | 5 | 1.000 | 0.923 | 0.960 |
| D2_Harshana | 115 | 91 | 3 | 21 | 0.968 | 0.812 | 0.883 |
| DnH_a | 93 | 10 | 0 | 83 | 1.000 | 0.108 | 0.194 |
| DnH_b | 93 | 11 | 3 | 79 | 0.786 | 0.122 | 0.212 |
| DnH_c | 93 | 7 | 0 | 86 | 1.000 | 0.075 | 0.140 |
| DnH_d | 93 | 20 | 1 | 72 | 0.952 | 0.217 | 0.354 |
| DnH_e | 93 | 22 | 1 | 70 | 0.957 | 0.239 | 0.383 |
| DnH_f | 93 | 45 | 1 | 47 | 0.978 | 0.489 | 0.652 |
| DnH_g | 93 | 22 | 0 | 71 | 1.000 | 0.237 | 0.383 |
| DnH_h | 93 | 37 | 0 | 56 | 1.000 | 0.398 | 0.569 |
| DnH_i | 93 | 39 | 0 | 54 | 1.000 | 0.419 | 0.591 |
| DnH_j | 93 | 46 | 2 | 45 | 0.958 | 0.505 | 0.662 |
| DnH_k | 93 | 47 | 1 | 45 | 0.979 | 0.511 | 0.671 |

Note: the per-identity table above is at the strict 0.80 threshold; the low DnH recall is
abstention, not misclassification, and rises sharply at 0.60–0.70 (see §6.2). At 0.70 the
global recall is 64% with precision 93%.

### 6.4 Confusion matrix interpretation
Errors are overwhelmingly **abstentions** (system declined to guess), not **misidentifications**.
The only cross-person leakage is a handful of faces predicted as `D2_Eni`. When the system
commits to a name, it is correct ~97% of the time at 0.80 (12 false matches in 1203).
This "fail-safe" behavior is exactly what a precision-first system should do.

---

## 7. Honest limitations

These are real and should be addressed before any security/operational claim:

1. **No open-set / impostor test.** Validation is **closed-set** — every test face belongs to
   someone in the gallery. We measured "is the right person identified," **not** "are strangers
   correctly rejected." A real precision claim requires an impostor set.
2. **D2_Abhinav has 0 test images** (his folder had no test split → all 11 went to train). He is
   enrolled but **unvalidated**, and 11 images is a thin reference.
3. **Train/inference preprocessing mismatch.** Training/gallery embeddings were built from plain
   aligned crops, but the live script applies far-face zoom/enhancement before embedding. Until
   these match, live far-face accuracy will be below the 0.94 benchmark.
4. **Embedding backbone runs on CPU.** DeepFace/TensorFlow has no native-Windows GPU support, so
   embedding extraction is CPU-bound. Detection (SCRFD/onnxruntime) and the adapter run on GPU.
5. **Close-up vs far-face domain gap.** The live `Known_faces` test uses close-up webcam faces,
   while the model was tuned on far faces — a different regime than the validation set.
6. **Small per-identity data.** DnH has 31 train images/person; robustness to new lighting/range
   is limited without augmentation or more data.

---

## 8. Recommendations / next steps

1. **Operating threshold:** 0.70 for balanced precision (93%/64%); 0.80 for maximum precision (97%/38%).
2. **Wire the adapter + gallery into the live script** (done in `___scrfd_zoom2.py`); ensure
   preprocessing matches training (limitation #3).
3. **Add an impostor/open-set test** to validate stranger rejection (limitation #1).
4. **Multi-image enrollment** per person (near/far, bright/dim) for a more robust gallery.
5. **Condition augmentation in training** (simulate distance via downscale+blur; lighting via
   gamma/brightness/contrast; sensor noise) and re-validate to improve range/lighting robustness.
6. **Give Abhinav a test split** so he can be measured (limitation #2).

---

## 9. Reproducibility

**Environment**
- Python 3.11.9 (venv at `faceenv/`)
- torch 2.11.0+cu128, torchvision 0.26.0+cu128 (CUDA 12.8, GPU verified)
- onnxruntime-gpu 1.19.2 (CUDA-12 build; uses PyTorch's bundled CUDA libs via `os.add_dll_directory`)
- insightface (SCRFD `buffalo_sc` for prep; `buffalo_l` for live), deepface + tf-keras (ArcFace), opencv-python, scikit-learn

**Run order**
```
python 00_prepare_data.py
python 01_finetune.py
python 02_build_gallery.py
python 03_validate.py
```

**Key artifact locations**
- Aligned crops / manifest: `../train_val/datasets/`
- Models (adapter, gallery, thresholds): `../train_val/models/`
- Reports & plots: `../train_val/reports/`

**Hyperparameters of record**
- Adapter: 512→256, BatchNorm, L2-norm. ArcFace loss s=32, m=0.50.
- Training: AdamW, lr 3e-4, cosine schedule, 40 epochs, batch 32.
- Gallery: top-80% by centroid cosine similarity.
- Validation: threshold sweep 0.10–0.99 (0.01 step); precision-max requires recall ≥ 0.30.

---

*All figures in this report are taken directly from `03_validate.py` output on the held-out
test set (1203 images). No numbers are estimated or projected.*
