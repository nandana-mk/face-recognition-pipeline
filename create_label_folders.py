"""Create per-person label subfolders inside each _unsorted/<video>/ folder."""
import os, re

BASE = r"C:\dev\face-recognition-pipeline\video_dataset\_unsorted"
TOKENS = {
    "sa": "Sakthi", "da": "Dakshina", "ab": "Abhinav", "na": "Nandana",
    "si": "Siddhartha", "ni": "Nitharshan", "bk": "Bk",
    "jaga": "Jaga", "harsh": "Harsh", "allen": "Allen",
}
DEFAULT = ["Sakthi", "Dakshina", "Abhinav", "Nandana", "Siddhartha", "Nitharshan", "Bk"]

if not os.path.isdir(BASE):
    print(f"[ERROR] not found: {BASE}"); raise SystemExit

for vid in sorted(os.listdir(BASE)):
    vpath = os.path.join(BASE, vid)
    if not os.path.isdir(vpath):
        continue
    toks = re.split(r"[^a-zA-Z0-9]+", vid.lower())
    people = []
    for t in toks:
        if t in TOKENS and TOKENS[t] not in people:
            people.append(TOKENS[t])
    if not people:
        people = list(DEFAULT)
    for p in people + ["_ignore"]:
        os.makedirs(os.path.join(vpath, p), exist_ok=True)
    print(f"{vid}  ->  {', '.join(people)}")
print("DONE")
