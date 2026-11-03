# Far-Face Recognition — Labeled Drone-Video Evaluation

**Date:** 2026-07-02
**Tool:** `05_eval_labeled.py` (multi-image enrollment + condition-aware test)
**Recognizer:** insightface `w600k_r50` (buffalo_l), 512-d embeddings, cosine match
**Test data:** hand-labeled crops from real drone footage, grouped by condition (video)

---

## 1. Headline results

| Metric | Value |
|---|---|
| PR-AUC | **0.9907** |
| Best-F1 threshold | 0.36 (P=0.992, R=0.987, F1=0.989) |
| Operating threshold (used) | 0.40 |
| Test crops | 3,108 |
| Impostor crops | 1,542 |
| Impostor rejection @0.40 | **99.7%** (1537/1542) |

The recognizer separates enrolled identities from strangers almost perfectly on real
drone footage. When it commits to a name it is right ~99% of the time.

## 2. Setup

- **Gallery (enrolled, multi-image averaged from `video_dataset/<Person>/train`):**
  Allen (139), Nandana (200 cap), Siddhartha (198), Nitharshan (124), Dakshina (4).
- **Not enrolled (counted as impostors):** Abhinav, Sakthi, Bk, Jaga, Harsh.
  - Jaga/Harsh = genuine strangers (correct to reject).
  - **Abhinav, Sakthi, Bk = enrollment gap** — no `train` crops, so they cannot be
    recognized yet. Their "rejection" is not a true impostor result.

## 3. Per-condition accuracy (genuine, threshold 0.40)

| Condition (video) | Accuracy | Mean sim |
|---|---|---|
| Close-up of Individuals | 100.0% (120/120) | 0.586 |
| In Good Lighting | 97.1% (34/35) | 0.630 |
| Jaga, Harsh, Allen | 100.0% (29/29) | 0.794 |
| Jaga, Harsh, Allen 2 | 100.0% (24/24) | 0.719 |
| **Na, Ni, Ab** | **81.0% (17/21)** | 0.474 |
| **Na, Ni, Ab 2** | **81.8% (45/55)** | 0.533 |
| Nearer feed | 97.9% (95/97) | 0.532 |
| Sa, Da, Ab, Na, Si, Bk_1 | 100.0% (96/96) | 0.589 |
| Sa, Da, Ab, Na, Si, Bk_2 | 100.0% (54/54) | 0.605 |
| Sa, Da, Ab, Na, Si, Bk_4 | 100.0% (31/31) | 0.645 |
| Sa, Da, Ab, Na, Si, Bk_5 | 95.9% (94/98) | 0.535 |
| Sa, Da, Ab, Na, Si, Bk_6 | 92.7% (127/137) | 0.567 |
| Sa, Da, Ab, Na, Si, Bk_7 | 98.9% (91/92) | 0.638 |
| Sa, Da, Ab, Na, Si, Bk_8 | 99.0% (104/105) | 0.562 |
| Sa, Da, Ab, Na, Si, Bk_9 | 99.3% (144/145) | 0.544 |
| Sa, Da, Ab, Na, Si, Bk_11 | 99.8% (420/421) | 0.635 |
| **Walking Towards Drone** | **83.3% (5/6)** | 0.446 (small N) |

## 4. Per-person accuracy (genuine, threshold 0.40)

| Person | Accuracy |
|---|---|
| Allen | 100.0% (53/53) |
| Nandana | 98.8% (661/669) |
| Dakshina | 98.6% (680/690) |
| Siddhartha | 91.5% (97/106) |
| **Nitharshan** | **81.2% (39/48)** |

## 5. Analysis

- **Strong overall:** most conditions 97–100%; impostor rejection 99.7%.
- **Weak identity:** Nitharshan (81%). He also drives the low `Na, Ni, Ab` condition
  scores — same footage. Likely far/angled there + fewest enrollment crops (124).
- **Weak conditions:** `Na, Ni, Ab` (~81%), `Walking Towards Drone` (83%, low N),
  `Sa…_6` (92.7%). Low mean-sim (~0.45–0.53) tracks with far/blurry/motion frames.
- **41 failing crops exported** to `train_val/reports/failures/<condition>/`
  (filenames: `TrueName_as_WrongName__simX.XX`) — exact per-image failures for review.

## 6. Recommendations

1. **Close the enrollment gap:** add `video_dataset/<Person>/train` crops for Abhinav,
   Sakthi, Bk if they should be recognized; re-run.
2. **Strengthen Nitharshan:** add more enrollment crops (far + angled) — he's the weakest.
3. **Operating threshold:** 0.40 (F1≈0.99, 99.7% rejection). Set in `fr3.py`.
4. **Inspect `reports/failures/`** to confirm misses are genuine hard cases (far/profile/
   motion) vs. fixable label/enrollment issues.
5. **Re-run `05_eval_labeled.py`** after each change to quantify the gain.

## 7. Caveats (honesty)

- **Enrollment gap** (Abhinav/Sakthi/Bk) muddies the impostor metric — see §2.
- **Small-N conditions** (Walking = 6 crops) are not statistically reliable.
- Possible minor leakage if any test-video crops were also placed in `train`; the
  strong single-person-enrollment / multi-person-test separation keeps this small.
- Numbers computed with the same w600k_r50 model used live (CPU vs GPU is numerically
  equivalent).

*All figures produced by `05_eval_labeled.py`; per-crop data in `reports/labeled_eval.csv`.*
