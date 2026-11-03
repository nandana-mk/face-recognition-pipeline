"""
15_calibrate_decision.py — pick the operating point with DATA, not a guessed 0.40.

Reads two existing reports (no GPU, no re-run needed):
  labeled_eval.csv        (cols: cond,true,pred,sim,impostor,path)  -> has IMPOSTORS -> measures PRECISION safety
  droneface_distance.csv  (cols: subject,dist_m,height_m,px,pred,sim,correct,accepted) -> has PIXELS -> measures RECALL headroom

Part A: sweep the global accept threshold on the impostor-bearing set -> Precision / Recall / F1 / impostor-rejection,
        and recommend the operating threshold (max F1 subject to impostor-rejection >= TARGET_IMP_REJ).
Part B: on the pixel-labelled set, show recall vs face-px and the exact recall you would GAIN by lowering the
        threshold to 0.35 (the [0.35,0.40) band), plus the near-range pitch collapse by camera height.

Why two files: DroneFace has pixels but no strangers (can't test precision); labeled_eval has strangers but no
pixel column (can't band by size). Together they bound the decision. To calibrate a *px-banded* threshold safely
you must first add a `px` column to 05_eval_labeled.py's output (see the note this script prints at the end).
"""
import os, csv
from collections import defaultdict

REPORTS = r"C:\dev\train_val\reports"
LABELED = os.path.join(REPORTS, "labeled_eval.csv")
DRONE   = os.path.join(REPORTS, "droneface_distance.csv")
TARGET_IMP_REJ = 0.99      # never accept more than 1% of strangers
MIN_PRECISION  = 0.99

def load(path):
    with open(path, newline='', encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

# ---------------- Part A: threshold sweep on impostor-bearing eval ----------------
def truthy(v): return str(v).strip().lower() in ('true', '1', 'yes')

rows = load(LABELED)
gen = [r for r in rows if not truthy(r['impostor'])]
imp = [r for r in rows if truthy(r['impostor'])]
print("=" * 74)
print(f"PART A · decision threshold sweep on labeled_eval.csv")
print(f"  genuine crops={len(gen)}  impostor crops={len(imp)}")
print("=" * 74)
print(f"  {'thr':>5} {'precision':>10} {'recall':>8} {'F1':>7} {'impostor-rej':>13} {'genFN':>7}")
best = None
for i in range(30, 56):
    t = i / 100.0
    tp = sum(1 for r in gen if float(r['sim']) >= t and r['pred'] == r['true'])
    fp_name = sum(1 for r in gen if float(r['sim']) >= t and r['pred'] != r['true'])
    fa_imp = sum(1 for r in imp if float(r['sim']) >= t)
    gen_fn = len(gen) - tp - fp_name            # genuine faces rejected (below thr)
    acc = tp + fp_name + fa_imp
    prec = tp / acc if acc else 1.0
    rec = tp / len(gen) if gen else 0.0
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    imp_rej = 1 - fa_imp / len(imp) if imp else 1.0
    flag = ""
    if imp_rej >= TARGET_IMP_REJ and prec >= MIN_PRECISION:
        if best is None or f1 > best[1]:
            best = (t, f1, prec, rec, imp_rej); flag = "  <- candidate"
    print(f"  {t:>5.2f} {prec:>10.4f} {rec:>8.4f} {f1:>7.4f} {imp_rej:>12.2%} {gen_fn:>7}{flag}")

if best:
    print(f"\n  RECOMMEND operating threshold = {best[0]:.2f}  "
          f"(F1={best[1]:.4f}  precision={best[2]:.4f}  recall={best[3]:.4f}  impostor-rej={best[4]:.2%})")
    print(f"  Constraint used: impostor-rejection >= {TARGET_IMP_REJ:.0%} AND precision >= {MIN_PRECISION:.0%}")
else:
    print("\n  [!] No threshold met the precision/impostor constraints — inspect the sweep above.")

# ---------------- Part B: pixel-labelled recall headroom + pitch collapse ----------------
d = load(DRONE)
for r in d:
    r['px'] = float(r['px']); r['sim'] = float(r['sim'])
    r['dist_m'] = float(r['dist_m']); r['height_m'] = float(r['height_m'])
    r['gt_correct'] = (r['pred'] == r['subject'])      # correct identity IGNORING threshold

print("\n" + "=" * 74)
print("PART B · recall headroom vs face-pixel size (droneface_distance.csv)")
print("=" * 74)
bins = [(0, 35), (35, 45), (45, 55), (55, 70), (70, 90), (90, 120), (120, 10000)]
labs = ['<35', '35-45', '45-55', '55-70', '70-90', '90-120', '120+']
print(f"  {'px band':>8} {'n':>5} {'recall@0.40':>12} {'recall@0.35':>12} {'mean-sim':>9}")
for (lo, hi), lab in zip(bins, labs):
    b = [r for r in d if lo <= r['px'] < hi]
    if not b: continue
    r40 = sum(1 for r in b if r['gt_correct'] and r['sim'] >= 0.40) / len(b)
    r35 = sum(1 for r in b if r['gt_correct'] and r['sim'] >= 0.35) / len(b)
    ms = sum(r['sim'] for r in b) / len(b)
    print(f"  {lab:>8} {len(b):>5} {r40:>11.1%} {r35:>11.1%} {ms:>9.3f}")

band = [r for r in d if 0.35 <= r['sim'] < 0.40]
band_ok = sum(1 for r in band if r['gt_correct'])
tot = len(d)
r40_all = sum(1 for r in d if r['gt_correct'] and r['sim'] >= 0.40) / tot
r35_all = sum(1 for r in d if r['gt_correct'] and r['sim'] >= 0.35) / tot
print(f"\n  Lowering 0.40 -> 0.35 adds {len(band)} accepts, of which {band_ok} are CORRECT "
      f"({band_ok/len(band):.0%} clean).")
print(f"  Overall recall {r40_all:.1%} -> {r35_all:.1%}   (+{(r35_all-r40_all)*100:.1f} pts)")

print("\n  Near-range recall by camera height (the pitch collapse; dist 2.0-3.0m):")
print(f"  {'height':>7} {'recall':>8} {'mean-sim':>9}")
for h in [1.5, 3.0, 4.0, 5.0]:
    b = [r for r in d if r['height_m'] == h and 2.0 <= r['dist_m'] <= 3.0]
    if not b: continue
    rc = sum(1 for r in b if r['gt_correct'] and r['sim'] >= 0.40) / len(b)
    ms = sum(r['sim'] for r in b) / len(b)
    print(f"  {h:>6.1f}m {rc:>8.1%} {ms:>9.3f}")

print("\n" + "-" * 74)
print("NEXT: to calibrate a PX-BANDED threshold safely (recall up on big faces, precision up on")
print("small ones) add two columns to 05_eval_labeled.py's CSV output: `px` (face size) and")
print("`sim2` (runner-up different-identity sim). Then this script can band by px AND test a margin")
print("gate against real impostors. Ask and I'll patch 05 + extend this calibrator.")
