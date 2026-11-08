import sys, json, numpy as np, torch, joblib
sys.path.append("../src")
from adapter import Adapter
from evalutil import open_set_metrics, pick_threshold
from sklearn.metrics import confusion_matrix

d = np.load("../data/features.npz", allow_pickle=True)
names = list(d["names"])
known_idx = [i for i, n in enumerate(names) if not n.startswith("stranger_")]
Xtr, ytr, Xte, yte = d["Xtr"], d["ytr"], d["Xte"], d["yte"]
results = {}

# Baseline: prototype matching on raw embeddings
def protos(X, y):
    P = np.stack([X[y == c].mean(0) for c in range(len(names))])
    return P / np.linalg.norm(P, axis=1, keepdims=True)

P0 = protos(Xtr, ytr)
sv = d["Xva"] @ P0.T
t0 = pick_threshold(sv.argmax(1), sv.max(1), d["yva"], known_idx)["t"]
st = Xte @ P0.T
results["baseline_prototypes"] = open_set_metrics(st.argmax(1), st.max(1), yte, known_idx, t0)

# Model A: logistic regression
a = joblib.load("../models/classifier_v1.joblib")
Pa = a["clf"].predict_proba(Xte)
results["A_logreg"] = open_set_metrics(a["clf"].classes_[Pa.argmax(1)], Pa.max(1), yte, known_idx, a["threshold"])

# Model B: ArcFace adapter
ck = torch.load("../models/adapter_v1.pt", map_location="cpu")
ad = Adapter()
ad.load_state_dict(ck["adapter"])
ad.eval()
with torch.no_grad():
    s = ad(torch.tensor(Xte)) @ ck["protos"].T
    conf, pred = s.max(1)
results["B_adapter"] = open_set_metrics(pred.numpy(), conf.numpy(), yte, known_idx, ck["threshold"])

print(json.dumps(results, indent=2))
json.dump(results, open("../eval/test_results.json", "w"), indent=2)

cm = confusion_matrix(yte, pred.numpy(), labels=range(len(names)))
np.savetxt("../eval/confusion_B.csv", cm, fmt="%d", delimiter=",")