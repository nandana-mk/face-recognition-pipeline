"""
STEP 0 — DATA EXTRACTION & FACE ALIGNMENT
==========================================
Extracts both zips, detects + aligns faces with SCRFD,
saves 112x112 aligned crops to:
  datasets/aligned/<identity>/<train|test>/<filename>.jpg

DnHFaces naming: <id>_gp_<group>_eo_<seq>.JPG
  - gp_0  → train
  - gp_3/4/5 → test

Data2 structure: Data2/<Person>/<train|test|flat>/<img>.jpg
  - Abhinav has no train/test subfolder → assigned to train
"""

import os
import sys
import zipfile
import shutil
import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import json

# ── Paths ───────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent                   # Facial_Recog/
DATASET_TV = BASE / "train_val" / "datasets"
RAW_DIR    = DATASET_TV / "raw"
ALIGNED_DIR= DATASET_TV / "aligned"

DNHFACES_ZIP = Path(r"C:\Users\Netrin\Downloads\References\Dataset and papers\DnHFaces.zip")
DATA2_ZIP    = Path(r"C:\Users\Netrin\Downloads\Data2-20260618T122935Z-3-001.zip")

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}

# ── SCRFD ────────────────────────────────────────────────────────────────────
print("[INFO] Loading SCRFD detector …")
detector = FaceAnalysis(
    name="buffalo_sc",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
detector.prepare(ctx_id=0, det_size=(640, 640))
print("[INFO] Detector ready\n")

# ── Helpers ──────────────────────────────────────────────────────────────────
def align_largest_face(img):
    """Detect, return aligned 112×112 crop of the largest face. None if no face."""
    faces = detector.get(img)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    return face_align.norm_crop(img, face.kps)

def save_aligned(img_path: Path, out_path: Path):
    """Load → align → save. Returns True on success."""
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    aligned = align_largest_face(img)
    if aligned is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), aligned)
    return True

# ── Extract zips ─────────────────────────────────────────────────────────────
def extract_zip(zip_path: Path, out_dir: Path):
    if out_dir.exists():
        print(f"  [SKIP] {out_dir} already exists")
        return
    print(f"  Extracting {zip_path.name} …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    print(f"  Done → {out_dir}")

print("=" * 60)
print("STEP 1: Extracting zips")
print("=" * 60)
extract_zip(DNHFACES_ZIP, RAW_DIR / "DnHFaces")
extract_zip(DATA2_ZIP,    RAW_DIR / "Data2")

# ── Process DnHFaces ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Aligning DnHFaces")
print("=" * 60)

# gp_0 → train;  gp_3/4/5 → test
TRAIN_GROUPS = {'0'}

photos_dir = RAW_DIR / "DnHFaces" / "open_data_set" / "photos_all"
if not photos_dir.exists():
    # Try finding it
    candidates = list((RAW_DIR / "DnHFaces").rglob("photos_all"))
    photos_dir = candidates[0] if candidates else photos_dir

stats_dnh = {'ok': 0, 'no_face': 0, 'err': 0}

for img_file in photos_dir.iterdir():
    if img_file.suffix not in VALID_EXT:
        continue

    name = img_file.stem          # e.g. ab_gp_3_eo_05
    parts = name.split('_')
    if len(parts) < 3 or parts[1] != 'gp':
        continue

    identity  = parts[0]          # e.g. "ab"
    group_num = parts[2]          # e.g. "3"
    split     = 'train' if group_num in TRAIN_GROUPS else 'test'

    out_path = ALIGNED_DIR / f"DnH_{identity}" / split / img_file.name

    if out_path.exists():
        stats_dnh['ok'] += 1
        continue

    ok = save_aligned(img_file, out_path)
    if ok:
        stats_dnh['ok'] += 1
    else:
        stats_dnh['no_face'] += 1

print(f"  DnHFaces: {stats_dnh['ok']} aligned, {stats_dnh['no_face']} no-face")

# ── Process Data2 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Aligning Data2")
print("=" * 60)

data2_root = RAW_DIR / "Data2" / "Data2"
if not data2_root.exists():
    candidates = list((RAW_DIR / "Data2").rglob("Data2"))
    data2_root = candidates[0] if candidates else data2_root

stats_d2 = {'ok': 0, 'no_face': 0}

for person_dir in data2_root.iterdir():
    if not person_dir.is_dir():
        continue
    person = person_dir.name

    # Check for train/test subdirs
    has_split = any((person_dir / s).is_dir() for s in ['train', 'test'])

    if has_split:
        for split in ['train', 'test']:
            split_dir = person_dir / split
            if not split_dir.exists():
                continue
            for img_file in split_dir.iterdir():
                if img_file.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}:
                    continue
                out_path = ALIGNED_DIR / f"D2_{person}" / split / img_file.name
                if out_path.exists():
                    stats_d2['ok'] += 1
                    continue
                ok = save_aligned(img_file, out_path)
                stats_d2['ok' if ok else 'no_face'] += 1
    else:
        # Flat dir (Abhinav) → all go to train
        for img_file in person_dir.iterdir():
            if img_file.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}:
                continue
            out_path = ALIGNED_DIR / f"D2_{person}" / "train" / img_file.name
            if out_path.exists():
                stats_d2['ok'] += 1
                continue
            ok = save_aligned(img_file, out_path)
            stats_d2['ok' if ok else 'no_face'] += 1

print(f"  Data2: {stats_d2['ok']} aligned, {stats_d2['no_face']} no-face")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

manifest = []
train_counts = {}
test_counts  = {}

for identity_dir in sorted(ALIGNED_DIR.iterdir()):
    if not identity_dir.is_dir():
        continue
    train_n = len(list((identity_dir / 'train').glob('*'))) if (identity_dir / 'train').exists() else 0
    test_n  = len(list((identity_dir / 'test').glob('*')))  if (identity_dir / 'test').exists()  else 0
    print(f"  {identity_dir.name:20s}  train={train_n:4d}  test={test_n:4d}")

    train_counts[identity_dir.name] = train_n
    test_counts[identity_dir.name]  = test_n

    for split in ['train', 'test']:
        split_dir = identity_dir / split
        if split_dir.exists():
            for f in split_dir.iterdir():
                manifest.append({'path': str(f), 'identity': identity_dir.name, 'split': split})

manifest_path = DATASET_TV / "manifest.json"
with open(manifest_path, 'w') as fh:
    json.dump(manifest, fh, indent=2)

total_train = sum(train_counts.values())
total_test  = sum(test_counts.values())
n_ids = len(train_counts)
print(f"\n  Identities: {n_ids}")
print(f"  Total train: {total_train}")
print(f"  Total test:  {total_test}")
print(f"\n  Manifest saved → {manifest_path}")
print("\n[DONE] Run 01_finetune.py next\n")
