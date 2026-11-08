import numpy as np, csv, cv2
from sklearn.metrics import roc_curve
import sys
sys.path.append("../src")
from faces import FaceEngine
import database as db
from matcher import FaceMatcher

con = db.connect()
engine = FaceEngine(det_size=(640, 640))
m = FaceMatcher(con, threshold=0.01) # threshold 0: we only want raw scores

scores, labels = [], []
# eval/test.csv format: path, truth (truth='unknown' for strangers)
for path, truth in csv.reader(open("../eval/test.csv")):
    img = cv2.imread(path)
    fs = engine.analyse(img)
    if len(fs) != 1:
        continue
    name, cat, score = m.match(fs[0]["embedding"])
    scores.append(score)
    labels.append(1 if (name == truth) else 0)

fpr, tpr, thr = roc_curve(labels, scores)
for t in np.arange(0.30, 0.71, 0.05):
    pred = np.array(scores) >= t
    lab = np.array(labels) == 1
    far = np.mean(pred[~lab]) if (~lab).any() else 0
    frr = np.mean(~pred[lab]) if lab.any() else 0
    print(f"threshold {t:.2f}: FAR {far:.3f} FRR {frr:.3f}")