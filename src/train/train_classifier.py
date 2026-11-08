import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from evalutil import open_set_metrics, pick_threshold

d = np.load("data/features.npz", allow_pickle=True)
names = list(d["names"])
known_idx = [i for i, n in enumerate(names) if not n.startswith("stranger_")]

clf = LogisticRegression(C=10, max_iter=3000, class_weight="balanced")
clf.fit(d["Xtr"], d["ytr"])

def run(X, y, t=None):
    P = clf.predict_proba(X)
    return clf.classes_[P.argmax(1)], P.max(1)

pred, conf = run(d["Xva"], d["yva"])
best = pick_threshold(pred, conf, d["yva"], known_idx, max_far=0.01)
print("threshold chosen on VALIDATION:", best)

joblib.dump({"clf": clf, "names": names, "threshold": best["t"]}, "models/classifier_v1.joblib")