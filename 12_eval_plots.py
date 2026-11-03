"""
12_eval_plots.py — turn the labeled evaluation into VISUALS.
Reads reports/labeled_eval.csv (written by 05_eval_labeled.py) and saves PNGs:
  confusion_matrix.png   — true identity (rows) vs predicted (cols, incl. Unknown)
  threshold_sweep.png    — precision / recall / F1 vs threshold
  pr_curve.png           — precision-recall curve (+ area)
  sim_distributions.png  — genuine vs impostor similarity histograms
  per_condition.png      — accuracy per condition (bar)
No GPU / model needed — pure analysis of the CSV.
"""
import os, csv, datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = r"C:\dev\train_val\reports\labeled_eval.csv"
# each run -> a fresh timestamped folder, so nothing is overwritten (no file-lock, no viewer cache)
OUT = os.path.join(r"C:\dev\train_val\reports\plots", datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S"))
THRESHOLD = 0.40
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(CSV):
    print(f"[ERROR] {CSV} not found — run 05_eval_labeled.py first."); raise SystemExit

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        rows.append({'cond': r['cond'], 'true': r['true'], 'pred': r['pred'],
                     'sim': float(r['sim']), 'impostor': r['impostor'] == 'True'})
print(f"[INFO] {len(rows)} crops loaded")

gen = [r for r in rows if not r['impostor']]
imp = [r for r in rows if r['impostor']]

# ---------- 1. threshold sweep + metrics ----------
def metrics(t):
    TP = FP = FN = TN = 0
    for r in rows:
        if r['impostor']:
            FP += r['sim'] >= t; TN += r['sim'] < t
        else:
            if r['sim'] >= t and r['pred'] == r['true']: TP += 1
            elif r['sim'] >= t: FP += 1
            else: FN += 1
    P = TP/(TP+FP+1e-9); R = TP/(TP+FN+1e-9)
    return P, R, 2*P*R/(P+R+1e-9)

ts = np.arange(0.20, 0.85, 0.02)
Ps, Rs, Fs = zip(*[metrics(t) for t in ts])

plt.figure(figsize=(8, 5))
plt.plot(ts, Ps, label='Precision', color='steelblue', lw=2)
plt.plot(ts, Rs, label='Recall', color='coral', lw=2)
plt.plot(ts, Fs, label='F1', color='green', lw=2)
plt.axvline(THRESHOLD, ls='--', color='gray', alpha=0.7, label=f'op threshold {THRESHOLD}')
plt.xlabel('similarity threshold'); plt.ylabel('score'); plt.ylim(0, 1.02)
plt.title('Precision / Recall / F1 vs threshold'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'threshold_sweep.png'), dpi=150); plt.close()

# ---------- 2. PR curve ----------
pairs = sorted(zip(Rs, Ps))
rr = [p[0] for p in pairs]; pp = [p[1] for p in pairs]
auc = float(np.trapezoid(pp, rr))
plt.figure(figsize=(6, 6))
plt.plot(rr, pp, color='steelblue', lw=2)
plt.fill_between(rr, pp, alpha=0.1, color='steelblue')
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.xlim(0, 1); plt.ylim(0, 1.02)
plt.title(f'Precision-Recall curve (AUC = {auc:.3f})'); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'pr_curve.png'), dpi=150); plt.close()

# ---------- 3. similarity distributions ----------
gen_sims = [r['sim'] for r in gen if r['pred'] == r['true']]
imp_sims = [r['sim'] for r in imp]
plt.figure(figsize=(8, 5))
plt.hist(gen_sims, bins=40, alpha=0.6, color='green', label=f'genuine (n={len(gen_sims)})', density=True)
plt.hist(imp_sims, bins=40, alpha=0.6, color='red', label=f'impostor (n={len(imp_sims)})', density=True)
plt.axvline(THRESHOLD, ls='--', color='black', label=f'threshold {THRESHOLD}')
plt.xlabel('cosine similarity'); plt.ylabel('density')
plt.title('Genuine vs impostor similarity (separation = quality)'); plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'sim_distributions.png'), dpi=150); plt.close()

# ---------- 4. per-condition accuracy bar ----------
from collections import defaultdict
byc = defaultdict(list)
for r in gen: byc[r['cond']].append(r)
conds = sorted(byc, key=lambda c: len(byc[c]), reverse=True)
accs = [sum(1 for r in byc[c] if r['sim'] >= THRESHOLD and r['pred'] == r['true'])/len(byc[c]) for c in conds]
plt.figure(figsize=(10, max(4, len(conds)*0.35)))
plt.barh([c[:34] for c in conds], accs, color='steelblue')
plt.xlabel(f'accuracy @ {THRESHOLD}'); plt.xlim(0, 1)
plt.title('Per-condition recognition accuracy'); plt.gca().invert_yaxis(); plt.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'per_condition.png'), dpi=150); plt.close()

# ---------- 5. confusion matrix ----------
trues = sorted(set(r['true'] for r in rows))
gal = sorted(set(r['pred'] for r in rows))
cols = gal + ['Unknown']
ci = {c: i for i, c in enumerate(cols)}; ri = {t: i for i, t in enumerate(trues)}
M = np.zeros((len(trues), len(cols)), int)
for r in rows:
    pc = r['pred'] if r['sim'] >= THRESHOLD else 'Unknown'
    M[ri[r['true']], ci[pc]] += 1
plt.figure(figsize=(max(7, len(cols)*0.8), max(5, len(trues)*0.5)))
plt.imshow(M, cmap='Blues')
plt.xticks(range(len(cols)), cols, rotation=45, ha='right', fontsize=8)
plt.yticks(range(len(trues)), trues, fontsize=8)
plt.xlabel('Predicted'); plt.ylabel('True')
plt.title(f'Confusion matrix @ {THRESHOLD}')
for i in range(len(trues)):
    for j in range(len(cols)):
        if M[i, j]:
            plt.text(j, i, M[i, j], ha='center', va='center', fontsize=7,
                     color='white' if M[i, j] > M.max()*0.5 else 'black')
plt.colorbar(fraction=0.03); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'confusion_matrix.png'), dpi=150); plt.close()

print(f"[DONE] saved 5 PNGs -> {OUT}")
for f in ['confusion_matrix', 'threshold_sweep', 'pr_curve', 'sim_distributions', 'per_condition']:
    print(f"  {f}.png")
