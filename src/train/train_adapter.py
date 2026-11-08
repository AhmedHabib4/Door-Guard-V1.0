import sys, numpy as np, torch, torch.nn.functional as F
sys.path.append("src")
from adapter import Adapter, ArcMargin
from evalutil import pick_threshold

torch.manual_seed(0)
d = np.load("data/features.npz", allow_pickle=True)
names = list(d["names"])
known_idx = [i for i, n in enumerate(names) if not n.startswith("stranger_")]

Xtr, ytr = torch.tensor(d["Xtr"]), torch.tensor(d["ytr"], dtype=torch.long)
Xva, yva = torch.tensor(d["Xva"]), torch.tensor(d["yva"], dtype=torch.long)

adapter, head = Adapter(), ArcMargin(512, len(names))
opt = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-2)

def prototypes(Z, y):
    return F.normalize(torch.stack([Z[y == c].mean(0) for c in range(len(names))]), dim=1)

best_score, patience, best_state = -1.0, 0, None

for epoch in range(200):
    adapter.train(); head.train()
    perm = torch.randperm(len(Xtr))
    total = 0.0
    for i in range(0, len(perm), 128):
        idx = perm[i:i + 128]
        loss = F.cross_entropy(head(adapter(Xtr[idx]), ytr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(idx)
        
    adapter.eval()
    with torch.no_grad():
        P = prototypes(adapter(Xtr), ytr)
        sims = adapter(Xva) @ P.T
        conf, pred = sims.max(1)
        pred, conf, y = pred.numpy(), conf.numpy(), yva.numpy()
        m = pick_threshold(pred, conf, y, known_idx, max_far=0.01)
        score = 1 - m["FRR"] if m else 0
        frr = m["FRR"] if m else "n/a"
        
    print(f"epoch {epoch:3d} loss {total / len(Xtr):.3f} val FRR@FAR<=1%: {frr}")
    
    if score > best_score:
        best_score, patience = score, 0
        best_state = ({k: v.clone() for k, v in adapter.state_dict().items()}, P.clone(), m)
    else:
        patience += 1
        if patience >= 15: # early stopping
            break

state, P, m = best_state
cats = [n.split("_", 1)[0] for n in names]
torch.save({"adapter": state, "protos": P, "names": names, "cats": cats, "threshold": m["t"]}, "models/adapter_v1.pt")
print("saved. validation result:", m)