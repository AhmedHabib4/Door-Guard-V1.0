import csv, glob, os, random
from collections import defaultdict

meta_rows = list(csv.DictReader(open("data/faces_meta.csv")))
video_of = {os.path.basename(r["file"]): r["video"] for r in meta_rows}
rows = []

for folder in sorted(glob.glob("data/labelled/*")):
    label = os.path.basename(folder)
    by_video = defaultdict(list)
    for f in glob.glob(folder + "/*.jpg"):
        by_video[video_of[os.path.basename(f)]].append(f)
        
    vids = sorted(by_video)
    random.Random(42).shuffle(vids)
    n = len(vids)
    
    if n < 3:
        print("WARNING: only", n, "videos for", label, "- all goes to train, cannot be evaluated")
        n_test, n_val = 0, 0
    else:
        n_test = max(1, round(0.20 * n))
        n_val = max(1, round(0.15 * n))
        
    test_v, val_v = set(vids[:n_test]), set(vids[n_test:n_test + n_val])
    
    for v in vids:
        split = "test" if v in test_v else "val" if v in val_v else "train"
        for f in by_video[v]:
            rows.append([f, label, v, split])

with open("data/splits.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "label", "video", "split"])
    w.writerows(rows)

count = defaultdict(lambda: defaultdict(int))
for _, label, _, split in rows:
    count[label][split] += 1

print(f"{'label':30s} train   val  test")
for label in sorted(count):
    c = count[label]
    print(f"{label:30s} {c['train']:5d} {c['val']:5d} {c['test']:5d}")