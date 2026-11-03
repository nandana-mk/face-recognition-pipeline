# Face Recognition Training & Validation Pipeline

Precision-focused training and validation for far-face recognition using **SCRFD + ArcFace**, optimized for drone/surveillance footage at 3–5m+ distance.

## Pipeline Overview

| Script | Purpose |
|--------|---------|
| `00_prepare_data.py` | Extract datasets → SCRFD align → 112×112 crops |
| `01_finetune.py` | Freeze ArcFace backbone → train metric adapter (ArcFace loss) |
| `02_build_gallery.py` | Build per-identity reference gallery from train split |
| `03_validate.py` | Full validation: precision/recall/F1, ROC, confusion matrix |

## Datasets

- **DnHFaces** (16 identities, ~2057 images): `gp_0` → train, `gp_3/4/5` → test
- **Data2** (Eni, Harshana, Abhinav — drone far-face crops): uses existing train/test split

Place zip files at:
```
C:\Users\Netrin\Downloads\References\Dataset and papers\DnHFaces.zip
C:\Users\Netrin\Downloads\Data2-20260618T122935Z-3-001.zip
```

## Setup

### Environment
```bash
# Uses faceenv (Python 3.11) — already configured in RUN_ALL.bat
# Required packages (all present in faceenv):
#   torch, insightface, deepface, opencv-python, numpy, matplotlib, scikit-learn
```

### Install on new machine (RTX 5090)
```bash
python -m venv faceenv
faceenv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install insightface deepface opencv-python numpy matplotlib scikit-learn onnxruntime-gpu
```

## Run

```bash
# Full pipeline (double-click or terminal):
RUN_ALL.bat

# Or step by step:
faceenv\Scripts\python.exe 00_prepare_data.py
faceenv\Scripts\python.exe 01_finetune.py
faceenv\Scripts\python.exe 02_build_gallery.py
faceenv\Scripts\python.exe 03_validate.py
```

## Architecture

```
ArcFace backbone (frozen, 512-dim)
        ↓
Linear(512 → 256) + BatchNorm  ← trainable adapter
        ↓
L2-normalize
        ↓
ArcFace loss (scale=32, margin=0.50)
```

Adapter fine-tuning adapts the embedding space to your specific camera/lighting
without overfitting on the small dataset (~2500 images, 19 identities).

## Outputs

After `03_validate.py`:
```
train_val/reports/
├── validation_report.txt    # Full metrics table + recommended threshold
├── pr_curve.png             # Adapted vs raw ArcFace PR curve
└── confusion_matrix.png     # Per-identity confusion at optimal threshold

train_val/models/
├── adapter_best.pt          # Best fine-tuned adapter weights
├── gallery.npz              # Per-identity mean embeddings
├── class_map.json           # Identity → class index mapping
└── best_thresholds.json     # Precision-max & F1-max thresholds
```

## Integration with scrfd_zoom2.py

After validation, update `SIMILARITY_THRESHOLD` in `___scrfd_zoom2.py` with the
recommended value printed at the end of `03_validate.py`.
