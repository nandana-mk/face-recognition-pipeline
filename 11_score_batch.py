"""
11_score_batch.py — turn batch_tracks.csv into precision / recall / F1.
Compares each video's ACCEPTED identities (decision != Unknown, track long enough)
against a ground-truth map of who is actually in each clip (from the filenames).
Video-level set scoring:
  TP = named someone who IS in the clip
  FP = named someone who is NOT in the clip (wrong / stranger matched)
  FN = a person in the clip that was missed
Edit GROUND_TRUTH rules if any video's contents differ.
"""
import csv, os
from collections import defaultdict

TRACKS_CSV = r"C:\dev\train_val\reports\batch_tracks.csv"
MIN_FRAMES = 5          # ignore flimsy tracks (fragmentation) — must persist >= this many frames

# who is ACTUALLY in each clip (enrolled names only; strangers -> not listed).
# Names must match the Known_faces stems: Abhinav1, Kailash, Tejeswar, etc.
def ground_truth(video):
    n = video.lower()
    if 'jaga' in n and 'allen' in n:      return {'Jaga', 'Harshana', 'Allen'}
    if 'allen' in n:                       return {'Allen'}
    if 'nandana' in n:                     return {'Nandana'}
    if 'siddhartha' in n:                  return {'Siddhartha'}
    if n.startswith('nitharshan'):         return {'Nitharshan'}
    if n.startswith('da.'):                return set()        # Dakshina — not enrolled
    if 'dark' in n:                        return {'Nandana', 'Abhinav1', 'Siddhartha', 'Nitharshan'}
    if 'na, ni, ab' in n:                  return {'Nandana', 'Nitharshan', 'Abhinav1'}
    if 'measured distance' in n:           return {'Abhinav1', 'Kailash', 'Tejeswar'}
    if 'sa, da, ab, na, si, bk' in n:      return {'Abhinav1', 'Nandana', 'Siddhartha'}
    return None    # unknown video -> skipped

# ---------- load tracks ----------
byvid = defaultdict(list)
with open(TRACKS_CSV) as fh:
    for r in csv.DictReader(fh):
        byvid[r['video']].append(r)

TP = FP = FN = 0
per_video = []
for vid, rows in byvid.items():
    gt = ground_truth(vid)
    if gt is None:
        continue
    accepted = [r for r in rows if r['decision'] != 'Unknown' and int(r['frames']) >= MIN_FRAMES]
    predicted = set(r['decision'] for r in accepted)
    tp = predicted & gt
    fp = predicted - gt
    fn = gt - predicted
    TP += len(tp); FP += len(fp); FN += len(fn)
    per_video.append((vid, gt, tp, fp, fn))

P = TP / (TP + FP + 1e-9); R = TP / (TP + FN + 1e-9); F1 = 2 * P * R / (P + R + 1e-9)

print("=" * 78)
print(f"BATCH SCORING  (video-level, tracks with >= {MIN_FRAMES} frames)")
print("=" * 78)
print(f"  Precision {P:.3f}   Recall {R:.3f}   F1 {F1:.3f}    (TP={TP} FP={FP} FN={FN})")
print("\n  video                                        expected           found            MISSED / WRONG")
print("  " + "-" * 92)
for vid, gt, tp, fp, fn in sorted(per_video):
    miss = f"miss:{','.join(sorted(fn))}" if fn else ""
    wrong = f"WRONG:{','.join(sorted(fp))}" if fp else ""
    flag = (miss + " " + wrong).strip() or "ok"
    print(f"  {vid[:44]:44s} {','.join(sorted(gt))[:18]:18s} {','.join(sorted(tp))[:16]:16s} {flag}")

# ---------- per-person criteria ----------
present, foundp, falsep = defaultdict(int), defaultdict(int), defaultdict(int)
for vid, gt, tp, fp, fn in per_video:
    for p in gt: present[p] += 1
    for p in tp: foundp[p] += 1
    for p in fp: falsep[p] += 1
print("\n  per-person:   present-in-clips   found(recall)   false-alarms")
for p in sorted(set(list(present) + list(falsep))):
    pr, fd, fa = present.get(p, 0), foundp.get(p, 0), falsep.get(p, 0)
    rec = f"{fd}/{pr} = {fd/pr:.0%}" if pr else f"0 (not in any clip)"
    print(f"  {p:12s}        {pr:>3}            {rec:<14}   {fa}")

n_videos = len(per_video)
print(f"\n  SUMMARY   videos scored: {n_videos}")
print(f"            Precision {P:.3f}  |  Recall {R:.3f}  |  F1 {F1:.3f}")
print(f"            correct IDs (TP) {TP}  ·  wrong names (FP) {FP}  ·  missed (FN) {FN}")

print("\nLegend: TP=correctly found · FP(WRONG)=named someone not in clip · FN(miss)=person in clip not found")
print("Tune MIN_FRAMES / the GROUND_TRUTH rules if a clip's real contents differ.")
