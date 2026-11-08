import csv, os, shutil, cv2, numpy as np
from sklearn.cluster import DBSCAN

E = np.load("data/emb_crops.npy")
meta = list(csv.DictReader(open("data/faces_meta.csv")))
EPS = 0.45 

labels = DBSCAN(eps=EPS, min_samples=6, metric="cosine").fit_predict(E)

def montage(files, cols=10, size=80, maxn=60):
    tiles = [cv2.resize(cv2.imread(f), (size, size)) for f in files[:maxn]]
    while len(tiles) % cols:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])

shutil.rmtree("data/clusters", ignore_errors=True)
print("cluster faces videos")

for c in sorted(set(labels), key=lambda c: (labels == c).sum()):
    name = "noise" if c == -1 else f"cluster_{c:03d}"
    idx = np.where(labels == c)[0]
    files = [meta[i]["file"] for i in idx]
    
    os.makedirs(f"data/clusters/{name}", exist_ok=True)
    for f in files:
        shutil.copy(f, f"data/clusters/{name}/{os.path.basename(f)}")
    cv2.imwrite(f"data/clusters/{name}_sheet.jpg", montage(files))
    print(f"{name:12s} {len(files):5d} {len(set(meta[i]['video'] for i in idx)):5d}")