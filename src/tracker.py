import math

class CentroidTracker:
    def __init__(self, max_dist=90, max_missing=8):
        self.next_id, self.objs = 0, {}
        self.max_dist, self.max_missing = max_dist, max_missing

    def update(self, boxes):
        results, used = [], set()
        for b in boxes:
            c = ((b[0]+b[2]) // 2, (b[1]+b[3]) // 2)
            best, bd = None, self.max_dist
            for oid, o in self.objs.items():
                if oid in used:
                    continue
                d = math.dist(c, o["c"])
                if d < bd:
                    best, bd = oid, d
            
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.objs[best] = {"c": c, "prev": c, "missing": 0}
            else:
                o = self.objs[best]
                o["prev"], o["c"], o["missing"] = o["c"], c, 0
            used.add(best)
            results.append((best, b))
            
        for oid in list(self.objs):
            if oid not in used:
                self.objs[oid]["missing"] += 1
                if self.objs[oid]["missing"] > self.max_missing:
                    del self.objs[oid]
        return results