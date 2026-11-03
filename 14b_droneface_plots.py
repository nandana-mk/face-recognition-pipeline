"""
14b_droneface_plots.py — full visual dashboard for the DroneFace distance benchmark.
Reads reports/droneface_distance.csv (produced by 14_droneface_distance.py) — NO GPU, fast,
re-run as often as you like. Produces a timestamped folder with a 6-panel dashboard PNG plus
standalone panels + a subject x distance recall heatmap.

Per-crop CSV columns: subject, dist_m, height_m, px, pred, sim, correct, accepted
  correct  = accepted AND predicted the right person   (True Positive)
  accepted = similarity >= threshold                   (model committed to a name)
Metrics per distance:
  recall    = correct / n                 (of all crops, how many correctly named)
  precision = correct / accepted          (of committed crops, how many right)
"""
import os, csv, datetime
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

CSV_PATH  = r"C:\dev\train_val\reports\droneface_distance.csv"
OUT_DIR   = r"C:\dev\train_val\reports"
THRESHOLD = 0.40

TEAL, AMBER, RED, GREY = '#0C7C88', '#B67611', '#C0392B', '#7f8c8d'

# ---------- load ----------
rows = []
with open(CSV_PATH, newline='') as fh:
    for r in csv.DictReader(fh):
        rows.append({'subject': r['subject'], 'dist': float(r['dist_m']),
                     'height': float(r['height_m']), 'px': float(r['px']),
                     'sim': float(r['sim']), 'correct': int(r['correct']),
                     'accepted': int(r['accepted'])})
print(f"[INFO] loaded {len(rows)} crops from {CSV_PATH}")

def agg(bucket):
    """return (recall, precision, mean_sim, mean_px, n) for a list of rows"""
    n = len(bucket)
    tp = sum(r['correct'] for r in bucket)
    ac = sum(r['accepted'] for r in bucket)
    recall = tp / n if n else 0.0
    prec = tp / ac if ac else np.nan          # NaN where nothing accepted (don't plot as 0)
    return recall, prec, np.mean([r['sim'] for r in bucket]), np.mean([r['px'] for r in bucket]), n

# ---------- by distance ----------
by_d = defaultdict(list)
for r in rows: by_d[r['dist']].append(r)
dists = sorted(by_d)
D_rec, D_prec, D_sim, D_px = [], [], [], []
for d in dists:
    rec, pr, ms, mp, _ = agg(by_d[d])
    D_rec.append(rec); D_prec.append(pr); D_sim.append(ms); D_px.append(mp)

# headline
gr, gp, gs, _, gn = agg(rows)
good = [d for d, a in zip(dists, D_rec) if a >= 0.90]
reliable = max(good) if good else None
best_i = int(np.argmax(D_rec))
print(f"[INFO] overall recall {gr:.1%}  precision {gp:.1%}  best@{dists[best_i]:.1f}m ({D_rec[best_i]:.0%})")

# ---------- by height ----------
by_h = defaultdict(list)
for r in rows: by_h[r['height']].append(r)
heights = sorted(by_h)
H_rec = [agg(by_h[h])[0] for h in heights]

# per-height recall-vs-distance lines
h_curves = {}
for h in heights:
    hd = defaultdict(list)
    for r in by_h[h]: hd[r['dist']].append(r)
    h_curves[h] = ([d for d in dists], [agg(hd[d])[0] if hd[d] else np.nan for d in dists])

# ---------- recall vs face pixel size ----------
px_bins = [0, 35, 45, 55, 70, 90, 120, 999]
px_lab = ['<35', '35-45', '45-55', '55-70', '70-90', '90-120', '120+']
px_buckets = defaultdict(list)
for r in rows:
    for bi in range(len(px_bins) - 1):
        if px_bins[bi] <= r['px'] < px_bins[bi + 1]:
            px_buckets[bi].append(r); break
PX_rec = [agg(px_buckets[bi])[0] if px_buckets[bi] else np.nan for bi in range(len(px_lab))]
PX_n   = [len(px_buckets[bi]) for bi in range(len(px_lab))]

# ---------- output folder ----------
pdir = os.path.join(OUT_DIR, "plots", datetime.datetime.now().strftime("droneface_dash_%H%M%S"))
os.makedirs(pdir, exist_ok=True)

def save(fig, name):
    fig.savefig(os.path.join(pdir, name), dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"[SAVED] {name}")

# ======================= 6-panel dashboard =======================
fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
fig.suptitle(f"DroneFace — recognition vs distance   (buffalo_l, thr={THRESHOLD}, "
             f"{gn} crops, 11 subjects)   overall P={gp:.0%} R={gr:.0%}",
             fontsize=13, fontweight='bold')

# A: recall + mean-sim vs distance
a = ax[0, 0]
a.plot(dists, D_rec, 'o-', color=TEAL, lw=2.2, label='recall')
a.plot(dists, D_sim, 's--', color=AMBER, lw=1.6, ms=4, label='mean similarity')
a.axhline(THRESHOLD, color=GREY, ls=':', label=f'threshold {THRESHOLD}')
if reliable: a.axvline(reliable, color=RED, ls='-', alpha=.5)
a.set_title('A · Recall & similarity vs distance'); a.set_xlabel('distance (m)')
a.set_ylabel('recall / similarity'); a.set_ylim(0, 1.02); a.grid(alpha=.3); a.legend(fontsize=8)

# B: precision vs distance
b = ax[0, 1]
b.plot(dists, D_prec, 'o-', color='#27ae60', lw=2.2)
b.set_title('B · Precision vs distance  (of accepted faces, % correct)')
b.set_xlabel('distance (m)'); b.set_ylabel('precision'); b.set_ylim(0, 1.05); b.grid(alpha=.3)
b.text(.5, .1, 'precision ≈ 100%: when it commits, it is right', transform=b.transAxes,
       ha='center', fontsize=9, color='#27ae60', style='italic')

# C: face pixel size vs distance
c = ax[0, 2]
c.plot(dists, D_px, 'o-', color='#8e44ad', lw=2.2)
c.axhline(45, color=GREY, ls=':', label='~45 px (recall collapses)')
c.set_title('C · Face resolution vs distance'); c.set_xlabel('distance (m)')
c.set_ylabel('face size (px)'); c.grid(alpha=.3); c.legend(fontsize=8)

# D: recall vs face pixel size
d = ax[1, 0]
xb = np.arange(len(px_lab))
bars = d.bar(xb, [0 if np.isnan(v) else v for v in PX_rec], color=TEAL, alpha=.85)
for i, (v, nn) in enumerate(zip(PX_rec, PX_n)):
    if not np.isnan(v): d.text(i, v + .02, f'{v:.0%}\nn={nn}', ha='center', fontsize=7)
d.set_xticks(xb); d.set_xticklabels(px_lab, fontsize=8)
d.set_title('D · Recall vs face size (px)'); d.set_xlabel('face size (px)')
d.set_ylabel('recall'); d.set_ylim(0, 1.05); d.grid(alpha=.3, axis='y')

# E: per-height recall vs distance
e = ax[1, 1]
cmap = plt.cm.viridis(np.linspace(0, .85, len(heights)))
for h, col in zip(heights, cmap):
    xs, ys = h_curves[h]
    e.plot(xs, ys, '-', color=col, lw=1.8, label=f'{h:g} m')
e.set_title('E · Recall vs distance, by camera height'); e.set_xlabel('distance (m)')
e.set_ylabel('recall'); e.set_ylim(0, 1.02); e.grid(alpha=.3); e.legend(title='height', fontsize=8)

# F: similarity distribution — correct vs wrong/rejected
f = ax[1, 2]
sim_ok = [r['sim'] for r in rows if r['correct']]
sim_no = [r['sim'] for r in rows if not r['correct']]
bins = np.linspace(0, 0.9, 40)
f.hist(sim_no, bins=bins, color=RED, alpha=.55, label=f'missed/rejected (n={len(sim_no)})')
f.hist(sim_ok, bins=bins, color=TEAL, alpha=.7, label=f'correct (n={len(sim_ok)})')
f.axvline(THRESHOLD, color='k', ls='--', label=f'threshold {THRESHOLD}')
f.set_title('F · Similarity distribution'); f.set_xlabel('cosine similarity')
f.set_ylabel('crops'); f.grid(alpha=.3); f.legend(fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.96])
save(fig, 'dashboard.png')

# ======================= standalone: headline curve =======================
fig2, a2 = plt.subplots(figsize=(9, 5))
a2.plot(dists, D_rec, 'o-', color=TEAL, lw=2.4, label='recall')
a2.plot(dists, D_sim, 's--', color=AMBER, lw=1.6, ms=4, label='mean similarity')
a2.axhline(THRESHOLD, color=GREY, ls=':', label=f'threshold {THRESHOLD}')
a2.axhline(0.90, color=RED, ls=':', alpha=.6, label='90% recall')
if reliable:
    a2.axvline(reliable, color=RED, ls='-', alpha=.4)
    a2.text(reliable, 1.0, f' reliable ≤ {reliable:g} m', color=RED, fontsize=9, va='top')
a2.set_title('DroneFace: recognition performance vs distance'); a2.set_xlabel('distance (m)')
a2.set_ylabel('recall / similarity'); a2.set_ylim(0, 1.04); a2.grid(alpha=.3); a2.legend()
save(fig2, 'recall_vs_distance.png')

# ======================= heatmap: subject x distance recall =======================
subs = sorted({r['subject'] for r in rows})
# group distances into coarser bands for readability
bands = [(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 17.5)]
band_lab = ['2-4', '4-6', '6-8', '8-10', '10-12', '12-14', '14-17']
M = np.full((len(subs), len(bands)), np.nan)
for si, s in enumerate(subs):
    for bi, (lo, hi) in enumerate(bands):
        bk = [r for r in rows if r['subject'] == s and lo <= r['dist'] < hi]
        if bk: M[si, bi] = agg(bk)[0]
fig3, a3 = plt.subplots(figsize=(9, 6))
im = a3.imshow(M, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
a3.set_xticks(range(len(band_lab))); a3.set_xticklabels(band_lab)
a3.set_yticks(range(len(subs))); a3.set_yticklabels(subs)
a3.set_xlabel('distance band (m)'); a3.set_ylabel('subject')
a3.set_title('Recall by subject × distance  (green=recognised, red=missed)')
for si in range(len(subs)):
    for bi in range(len(bands)):
        if not np.isnan(M[si, bi]):
            a3.text(bi, si, f'{M[si, bi]:.0%}', ha='center', va='center', fontsize=7,
                    color='black')
fig3.colorbar(im, ax=a3, label='recall'); save(fig3, 'subject_distance_heatmap.png')

# ======================= per-height summary bar =======================
fig4, a4 = plt.subplots(figsize=(6, 4.5))
a4.bar([f'{h:g} m' for h in heights], H_rec, color=cmap)
for i, v in enumerate(H_rec): a4.text(i, v + .01, f'{v:.0%}', ha='center', fontsize=9)
a4.set_title('Overall recall by camera height'); a4.set_ylabel('recall')
a4.set_ylim(0, max(H_rec) * 1.2); a4.grid(alpha=.3, axis='y'); save(fig4, 'recall_by_height.png')

print(f"\n[DONE] dashboard + 4 standalone plots -> {pdir}")
print("  dashboard.png  recall_vs_distance.png  subject_distance_heatmap.png  recall_by_height.png")
