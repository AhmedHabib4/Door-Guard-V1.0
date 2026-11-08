import numpy as np

def open_set_metrics(pred, conf, y, known_idx, t):
    known_idx = np.array(known_idx)
    is_known = np.isin(y, known_idx)
    accepted = (conf >= t) & np.isin(pred, known_idx)
    correct = accepted & (pred == y)
    
    frr = 1 - correct[is_known].mean()
    far = accepted[~is_known].mean() if (~is_known).any() else 0.0
    misid = (accepted & (pred != y))[is_known].mean()
    
    return {"t": round(float(t), 3), "FRR": round(float(frr), 4),
            "FAR_stranger": round(float(far), 4), "misID_known": round(float(misid), 4)}

def pick_threshold(pred, conf, y, known_idx, max_far=0.01):
    best = None
    for t in np.linspace(0.05, 0.99, 95):
        m = open_set_metrics(pred, conf, y, known_idx, t)
        if m["FAR_stranger"] <= max_far and (best is None or m["FRR"] < best["FRR"]):
            best = m
    return best