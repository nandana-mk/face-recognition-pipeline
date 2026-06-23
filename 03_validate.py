"""
STEP 3 — VALIDATION WITH PRECISION-FOCUSED METRICS
===================================================
Runs the full pipeline on the test split:
  aligned crop → ArcFace backbone → adapter projection → gallery match

Reports:
  - ROC curve (AUC)
  - Precision / Recall / F1 at every threshold
  - Precision-maximizing threshold
  - Per-identity breakdown
  - Confusion matrix
  - Comparison: raw ArcFace vs fine-tuned adapter

Output: train_val/reports/validation_report.txt
        train_val/reports/roc_curve.png
        train_val/reports/confusion_matrix.png
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

BASE        = Path(__file__).parent.parent
DATASET_TV  = BASE / "train_val" / "datasets"
MODEL_DIR   = BASE / "train_val" / "models"
REPORT_DIR  = BASE / "train_val" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST    = DATASET_TV / "manifest.json"
CLASS_MAP_F = MODEL_DIR / "class_map.json"
ADAPTER_PT  = MODEL_DIR / "adapter_best.pt"
GALLERY_F   = MODEL_DIR / "gallery.npz"

# ── PyTorch + adapter ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
except ImportError:
    print("[ERROR] PyTorch required.")
    sys.exit(1)

with open(CLASS_MAP_F) as fh:
    meta = json.load(fh)
EMBED_DIM = meta['embed_dim']
IN_DIM    = meta['in_dim']
class_map = meta['class_map']

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
adapter.load_state_dict(torch.load(ADAPTER_PT, map_location=DEVICE))
adapter.eval()

def adapt_batch(raw_embs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.from_numpy(raw_embs).float().to(DEVICE)
        return adapter(t).cpu().numpy()

# ── Load gallery ──────────────────────────────────────────────────────────────
gal         = np.load(GALLERY_F, allow_pickle=True)
gal_names   = list(gal['names'])
gal_embs    = gal['embeddings']     # (C, EMBED_DIM) — adapted
gal_raw     = gal['raw_embeddings'] # (C, 512) — raw backbone

print(f"[INFO] Gallery: {len(gal_names)} identities")

# ── Load DeepFace ArcFace ─────────────────────────────────────────────────────
print("[INFO] Loading ArcFace backbone …")
os.environ['DEEPFACE_HOME'] = str(BASE)
from deepface import DeepFace

def get_raw_embedding(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.shape[:2] != (112, 112):
        img_bgr = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_LANCZOS4)
    emb = DeepFace.represent(
        img_path=img_bgr,
        model_name="ArcFace",
        detector_backend="skip",
        enforce_detection=False
    )[0]["embedding"]
    v = np.array(emb, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)

print("[INFO] Backbone ready\n")

# ── Matching helpers ───────────────────────────────────────────────────────────
def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def match_adapted(emb_proj: np.ndarray):
    """Match projected embedding against gallery. Returns (name, similarity)."""
    sims  = gal_embs @ emb_proj           # (C,)
    idx   = np.argmax(sims)
    return gal_names[idx], float(sims[idx])

def match_raw(emb_raw: np.ndarray):
    """Match raw backbone embedding against raw gallery."""
    sims  = gal_raw @ emb_raw
    idx   = np.argmax(sims)
    return gal_names[idx], float(sims[idx])

# ── Process test set ──────────────────────────────────────────────────────────
print("[INFO] Processing test images …\n")

with open(MANIFEST) as fh:
    manifest = json.load(fh)
test_items = [m for m in manifest if m['split'] == 'test']

results = []   # list of dicts per test image

for i, item in enumerate(test_items):
    p = Path(item['path'])
    if not p.exists():
        continue

    img = cv2.imread(str(p))
    if img is None:
        continue

    raw = get_raw_embedding(img)

    # Adapted match
    proj  = adapt_batch(raw[np.newaxis])[0]
    name_a, sim_a = match_adapted(proj)

    # Raw match (baseline)
    name_r, sim_r = match_raw(raw)

    results.append({
        'true_id'    : item['identity'],
        'pred_adapted': name_a,
        'sim_adapted' : sim_a,
        'pred_raw'    : name_r,
        'sim_raw'     : sim_r,
    })

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(test_items)} …")

print(f"[INFO] {len(results)} test images processed\n")

# ── Metric calculation at threshold ───────────────────────────────────────────
def compute_metrics(results, threshold, sim_key='sim_adapted', pred_key='pred_adapted'):
    """
    TP: predicted known identity == true identity AND sim >= threshold
    FP: predicted known identity != true identity AND sim >= threshold
    FN: sim < threshold  (system said "unknown" or wrong, but should have been identified)
    TN: sim < threshold  AND true_id not in gallery (shouldn't occur since all test IDs are known)
    """
    TP = FP = FN = TN = 0
    for r in results:
        sim  = r[sim_key]
        pred = r[pred_key]
        true = r['true_id']
        if sim >= threshold:
            if pred == true:
                TP += 1
            else:
                FP += 1
        else:
            # System abstains (unknown)
            FN += 1   # missed a known person

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return {'TP': TP, 'FP': FP, 'FN': FN,
            'precision': precision, 'recall': recall, 'f1': f1}

# ── Sweep thresholds ──────────────────────────────────────────────────────────
thresholds = np.arange(0.10, 0.99, 0.01)

rows_adapted = []
rows_raw     = []

for t in thresholds:
    m_a = compute_metrics(results, t, 'sim_adapted', 'pred_adapted')
    m_r = compute_metrics(results, t, 'sim_raw',     'pred_raw')
    m_a['threshold'] = t
    m_r['threshold'] = t
    rows_adapted.append(m_a)
    rows_raw.append(m_r)

# ── Find precision-maximizing threshold ───────────────────────────────────────
def best_precision_threshold(rows, min_recall=0.30):
    """Find threshold with highest precision where recall >= min_recall."""
    candidates = [r for r in rows if r['recall'] >= min_recall]
    if not candidates:
        candidates = rows
    return max(candidates, key=lambda r: r['precision'])

best_a = best_precision_threshold(rows_adapted)
best_r = best_precision_threshold(rows_raw)

# Also find best F1
best_f1_a = max(rows_adapted, key=lambda r: r['f1'])
best_f1_r = max(rows_raw,     key=lambda r: r['f1'])

# ── Per-identity metrics (at best threshold) ──────────────────────────────────
def per_identity_metrics(results, threshold, sim_key='sim_adapted', pred_key='pred_adapted'):
    ids = sorted({r['true_id'] for r in results})
    out = {}
    for identity in ids:
        sub = [r for r in results if r['true_id'] == identity]
        TP = sum(1 for r in sub if r[sim_key] >= threshold and r[pred_key] == identity)
        FP = sum(1 for r in sub if r[sim_key] >= threshold and r[pred_key] != identity)
        FN = sum(1 for r in sub if r[sim_key] <  threshold)
        p  = TP / (TP + FP + 1e-9)
        rc = TP / (TP + FN + 1e-9)
        f1 = 2 * p * rc / (p + rc + 1e-9)
        out[identity] = {'n': len(sub), 'TP': TP, 'FP': FP, 'FN': FN,
                         'precision': p, 'recall': rc, 'f1': f1}
    return out

per_id = per_identity_metrics(results, best_a['threshold'])

# ── AUC (area under precision-recall curve) ───────────────────────────────────
def pr_auc(rows):
    ps = [r['precision'] for r in rows]
    rs = [r['recall']    for r in rows]
    # Sort by recall
    pairs = sorted(zip(rs, ps))
    rs_s  = [p[0] for p in pairs]
    ps_s  = [p[1] for p in pairs]
    return float(np.trapz(ps_s, rs_s))

auc_a = pr_auc(rows_adapted)
auc_r = pr_auc(rows_raw)

# ── Confusion matrix ──────────────────────────────────────────────────────────
def build_confusion(results, threshold, sim_key, pred_key):
    ids      = sorted({r['true_id'] for r in results})
    id2i     = {n: i for i, n in enumerate(ids)}
    n        = len(ids)
    C        = np.zeros((n, n + 1), dtype=int)   # +1 col for "abstain"

    for r in results:
        i = id2i[r['true_id']]
        if r[sim_key] >= threshold:
            j = id2i.get(r[pred_key], n)   # n = "wrong not in ids"
        else:
            j = n   # abstain column
        if j < n + 1:
            C[i, j] += 1
    return C, ids

C_a, ids = build_confusion(results, best_a['threshold'], 'sim_adapted', 'pred_adapted')

# ── Write report ──────────────────────────────────────────────────────────────
report_path = REPORT_DIR / "validation_report.txt"

lines = []
def h(s):  lines.append(s)

h("=" * 70)
h("FACE RECOGNITION VALIDATION REPORT")
h(f"Test images: {len(results)}")
h("=" * 70)

h("\n── ADAPTED MODEL (fine-tuned) ─────────────────────────────────────────")
h(f"  PR-AUC                : {auc_a:.4f}")
h(f"  Best PRECISION thresh : {best_a['threshold']:.2f}  →  P={best_a['precision']:.4f}  R={best_a['recall']:.4f}  F1={best_a['f1']:.4f}")
h(f"  Best F1 thresh        : {best_f1_a['threshold']:.2f}  →  P={best_f1_a['precision']:.4f}  R={best_f1_a['recall']:.4f}  F1={best_f1_a['f1']:.4f}")
h(f"  TP={best_a['TP']}  FP={best_a['FP']}  FN={best_a['FN']}")

h("\n── RAW ARCFACE (baseline) ─────────────────────────────────────────────")
h(f"  PR-AUC                : {auc_r:.4f}")
h(f"  Best PRECISION thresh : {best_r['threshold']:.2f}  →  P={best_r['precision']:.4f}  R={best_r['recall']:.4f}  F1={best_r['f1']:.4f}")
h(f"  Best F1 thresh        : {best_f1_r['threshold']:.2f}  →  P={best_f1_r['precision']:.4f}  R={best_f1_r['recall']:.4f}  F1={best_f1_r['f1']:.4f}")

h("\n── PER-IDENTITY (at precision-max threshold) ──────────────────────────")
h(f"  {'Identity':30s}  {'N':>4}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
h("  " + "-" * 70)
for idn, m in sorted(per_id.items()):
    h(f"  {idn:30s}  {m['n']:>4}  {m['TP']:>4}  {m['FP']:>4}  {m['FN']:>4}  {m['precision']:>6.3f}  {m['recall']:>6.3f}  {m['f1']:>6.3f}")

h("\n── THRESHOLD TABLE (adapted) ──────────────────────────────────────────")
h(f"  {'Thresh':>7}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
h("  " + "-" * 50)
for r in rows_adapted[::5]:   # every 5th row to keep compact
    h(f"  {r['threshold']:>7.2f}  {r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}  {r['TP']:>5}  {r['FP']:>5}  {r['FN']:>5}")

h("\n── CONFUSION MATRIX (adapted, rows=true, cols=predicted) ─────────────")
header = "  " + " ".join(f"{n[:8]:>8}" for n in ids) + "  [abstain]"
h(header)
for i, idn in enumerate(ids):
    row_str = "  ".join(f"{C_a[i, j]:>8}" for j in range(len(ids)))
    h(f"  {idn[:12]:12s}  {row_str}  {C_a[i, len(ids)]:>8}")

report_text = "\n".join(lines)
print(report_text)

with open(report_path, 'w') as fh:
    fh.write(report_text)
print(f"\n[SAVED] {report_path}")

# ── Plot ROC / PR curve ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Face Recognition Validation", fontsize=14)

    # Precision-Recall
    ax = axes[0]
    for rows, label, color in [(rows_adapted, f'Adapted (AUC={auc_a:.3f})', 'steelblue'),
                                (rows_raw,     f'Raw ArcFace (AUC={auc_r:.3f})', 'coral')]:
        rs = [r['recall']    for r in rows]
        ps = [r['precision'] for r in rows]
        pairs = sorted(zip(rs, ps))
        ax.plot([p[0] for p in pairs], [p[1] for p in pairs], label=label, color=color, lw=2)

    ax.axvline(best_a['recall'],    color='steelblue', ls='--', alpha=0.6, label=f"Best-P thresh={best_a['threshold']:.2f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # F1 vs threshold
    ax = axes[1]
    for rows, label, color in [(rows_adapted, 'Adapted', 'steelblue'),
                                (rows_raw,     'Raw ArcFace', 'coral')]:
        ts = [r['threshold'] for r in rows]
        f1 = [r['f1']        for r in rows]
        ax.plot(ts, f1, label=label, color=color, lw=2)

    ax.axvline(best_a['threshold'],    color='steelblue', ls='--', alpha=0.6,
               label=f"Best-P @{best_a['threshold']:.2f}")
    ax.axvline(best_f1_a['threshold'], color='green',     ls='--', alpha=0.6,
               label=f"Best-F1 @{best_f1_a['threshold']:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1")
    ax.set_title("F1 Score vs Threshold"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    roc_path = REPORT_DIR / "pr_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    print(f"[SAVED] {roc_path}")

    # Confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(ids)), max(6, len(ids) - 2)))
    col_labels = ids + ['abstain']
    im = ax.imshow(C_a, cmap='Blues')
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(len(ids)));         ax.set_yticklabels(ids, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (threshold={best_a['threshold']:.2f})")
    plt.colorbar(im, ax=ax)
    for i in range(len(ids)):
        for j in range(len(col_labels)):
            ax.text(j, i, str(C_a[i, j]), ha='center', va='center', fontsize=6,
                    color='white' if C_a[i, j] > C_a.max() * 0.5 else 'black')
    plt.tight_layout()
    cm_path = REPORT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[SAVED] {cm_path}")

except ImportError:
    print("[WARN] matplotlib not found — skipping plots. pip install matplotlib")

# ── Save best thresholds for production use ───────────────────────────────────
best_thresh_file = MODEL_DIR / "best_thresholds.json"
with open(best_thresh_file, 'w') as fh:
    json.dump({
        'adapted_precision_max': {
            'threshold': float(best_a['threshold']),
            'precision': float(best_a['precision']),
            'recall':    float(best_a['recall']),
            'f1':        float(best_a['f1']),
        },
        'adapted_f1_max': {
            'threshold': float(best_f1_a['threshold']),
            'precision': float(best_f1_a['precision']),
            'recall':    float(best_f1_a['recall']),
            'f1':        float(best_f1_a['f1']),
        },
        'raw_precision_max': {
            'threshold': float(best_r['threshold']),
            'precision': float(best_r['precision']),
        },
        'pr_auc_adapted': float(auc_a),
        'pr_auc_raw':     float(auc_r),
    }, fh, indent=2)
print(f"[SAVED] {best_thresh_file}")
print("\n[DONE] Validation complete!\n")
print("=" * 70)
print(f"  RECOMMENDED THRESHOLD (max precision): {best_a['threshold']:.2f}")
print(f"  → Precision: {best_a['precision']:.4f}  Recall: {best_a['recall']:.4f}  F1: {best_a['f1']:.4f}")
print("=" * 70)
print(f"\nUpdate SIMILARITY_THRESHOLD in scrfd_zoom2.py to: {best_a['threshold']:.2f}")
print()
