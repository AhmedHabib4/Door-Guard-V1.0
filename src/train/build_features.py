import csv, numpy as np, cv2
from embed import embed_batch

rng = np.random.default_rng(0)

def augment(img):
    out = img.copy()
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    # contrast and brightness
    a, b = rng.uniform(0.6, 1.4), rng.uniform(-30, 30)
    out = np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    # far-away/low-resolution face
    if rng.random() < 0.4:
        s = int(rng.integers(40, 90))
        out = cv2.resize(cv2.resize(out, (s, s)), (112, 112))
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    # infrared night look
    if rng.random() < 0.3:
        out = cv2.cvtColor(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    if rng.random() < 0.3:
        out = np.clip(out.astype(np.float32) + rng.normal(0, 8, out.shape), 0, 255).astype(np.uint8)
    return out

rows = list(csv.DictReader(open("data/splits.csv")))
names = sorted(set(r["label"] for r in rows))
idx = {n: i for i, n in enumerate(names)}
N_AUG = 3

def build(split, aug):
    X, y = [], []
    sel = [r for r in rows if r["split"] == split]
    for i in range(0, len(sel), 64):
        batch = sel[i:i + 64]
        imgs = [cv2.imread(r["file"]) for r in batch]
        X.append(embed_batch(imgs))
        y += [idx[r["label"]] for r in batch]
        for _ in range(aug): # augmented copies per training image
            X.append(embed_batch([augment(im) for im in imgs]))
            y += [idx[r["label"]] for r in batch]
    return np.concatenate(X), np.array(y)

Xtr, ytr = build("train", N_AUG)
Xva, yva = build("val", 0)
Xte, yte = build("test", 0)

np.savez("data/features.npz", Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte, names=np.array(names))
print("train", Xtr.shape, "val", Xva.shape, "test", Xte.shape)