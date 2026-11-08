import numpy as np
import torch
from collections import defaultdict, deque, Counter
from adapter import Adapter

class FaceMatcher:
    def __init__(self, con, threshold=0.45):
        rows = con.execute("SELECT p.name, p.category, e.embedding FROM face_embeddings e JOIN people p ON p.id = e.person_id").fetchall()
        self.names = [r[0] for r in rows]
        self.cats = [r[1] for r in rows]
        self.E = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 512), np.float32)
        self.threshold = threshold

    def match(self, emb):
        if len(self.E) == 0:
            return "unknown", "unknown", 0.0
        sims = self.E @ emb
        best_name, best_score = "unknown", -1.0
        for name in set(self.names):
            idx = [i for i, n in enumerate(self.names) if n == name]
            score = float(np.sort(sims[idx])[-3:].mean())
            if score > best_score:
                best_name, best_score = name, score
        if best_score >= self.threshold:
            return best_name, self.cats[self.names.index(best_name)], best_score
        return "unknown", "unknown", best_score

class TrainedMatcher:
    def __init__(self, path="../models/adapter_v1.pt"):
        ck = torch.load(path, map_location="cpu")
        self.adapter = Adapter()
        self.adapter.load_state_dict(ck["adapter"])
        self.adapter.eval()
        self.protos = ck["protos"]
        self.full = ck["names"]
        self.cats = [n.split("_", 1)[0] for n in self.full]
        self.names = [n.split("_", 1)[1].replace("_", "") for n in self.full]
        self.threshold = ck["threshold"]

    @torch.no_grad()
    def match(self, emb):
        z = self.adapter(torch.from_numpy(emb)[None])
        sims = (z @ self.protos.T)[0]
        i = int(sims.argmax())
        score = float(sims[i])
        if self.cats[i] == "stranger" or score < self.threshold:
            return "unknown", "unknown", score
        return self.names[i], self.cats[i], score

class IdentityVoter:
    def __init__(self, window=8, votes_needed=3):
        self.hist = defaultdict(lambda: deque(maxlen=window))
        self.votes_needed = votes_needed

    def add(self, key, name):
        self.hist[key].append(name)

    def decide(self, key):
        names = [n for n in self.hist[key] if n != "unknown"]
        if not names:
            return None
        name, count = Counter(names).most_common(1)[0]
        return name if count > self.votes_needed else None