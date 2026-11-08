import time, os, cv2, yaml, datetime
import database as db
from camera import CameraStream
from motion import MotionDetector
from detector import ObjectDetector
from faces import FaceEngine
from matcher import TrainedMatcher, IdentityVoter
from tracker import CentroidTracker
from rules import RULES, build_message, is_unusual
from notifier import notify

cfg = yaml.safe_load(open("../config.yaml"))
P = cfg["processing"]
con = db.connect()

detector = ObjectDetector(size=P["yolo_size"])
faces = FaceEngine()
matcher = TrainedMatcher("../models/adapter_v1.pt") # Ensure you trained this in T7!
voter = IdentityVoter(votes_needed=P["votes_needed"])
os.makedirs("../snapshots", exist_ok=True)

class Cooldown:
    def __init__(self, s): 
        self.s, self.last = s, {}
    def ok(self, k):
        n = time.time()
        if n - self.last.get(k, 0) >= self.s:
            self.last[k] = n
            return True
        return False

cooldown = Cooldown(P["alert_cooldown_seconds"])

def side(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

def snapshot(frame, box, tag):
    x1, y1, x2, y2 = box
    img = frame.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    path = f"../snapshots/{datetime.datetime.now():%Y%m%d_%H%M%S}_{tag}.jpg"
    cv2.imwrite(path, img)
    return path

def raise_event(cam, kind, identity, category, score, frame, box):
    key = (cam, identity, kind)
    if not cooldown.ok(key): return
    
    ts = f"{datetime.datetime.now():%H:%M:%S}"
    path = snapshot(frame, box, kind)
    db.log_event(con, cam, kind, identity, category, score, path)
    
    rule = RULES.get(category, RULES["unknown"])
    now = datetime.datetime.now()
    text = build_message(identity, category, kind, cam, ts)
    urgent = (kind == "unknown" or is_unusual(category, now.weekday(), now.hour))
    
    if urgent: text = "ALERT: " + text
    if urgent or rule.get("alert") != "quiet" or kind == "enter":
        notify(text, path, silent=(rule.get("alert") == "quiet" and not urgent))

def run():
    cams = {c["name"]: CameraStream(c["name"], c["rtsp_sub"]) for c in cfg["cameras"]}
    state = {c["name"]: {"motion": MotionDetector(), "tracker": CentroidTracker(),
                         "first_seen": {}, "decided": {}, "last_motion": 0,
                         "line": c["entry_line"]} for c in cfg["cameras"]}
    interval = 1.0 / P["fps"]
    
    while True:
        t0 = time.time()
        for name, cam in cams.items():
            frame = cam.read()
            if frame is None: continue
            
            st = state[name]
            if st["motion"].detect(frame):
                st["last_motion"] = time.time()
                
            if time.time() - st["last_motion"] > 4:
                st["tracker"].update([])
                continue # No motion, sleep camera
                
            dets = detector.detect(frame)
            tracked = st["tracker"].update([d["box"] for d in detector.persons(dets)])
            
            for tid, box in tracked:
                key = (name, tid)
                st["first_seen"].setdefault(tid, time.time())
                x1, y1, x2, y2 = box
                h = y2 - y1
                crop = frame[max(0, y1 - 10):y1 + int(h * 0.6), max(0, x1 - 10):x2 + 10]
                
                if crop.size and tid not in st["decided"]:
                    for f in faces.analyse(crop):
                        if f["size"] >= 50 and f["det_score"] >= 0.6:
                            who, cat, score = matcher.match(f["embedding"])
                            voter.add(key, who)
                            
                    who = voter.decide(key)
                    if who:
                        cat = matcher.cats[matcher.names.index(who)]
                        st["decided"][tid] = (who, cat)
                        raise_event(name, "visit", who, cat, 1.0, frame, box)
                    elif time.time() - st["first_seen"][tid] > P["unknown_after_seconds"]:
                        st["decided"][tid] = ("unknown", "unknown")
                        raise_event(name, "unknown", "unknown", "unknown", 0.0, frame, box)
                        
                o = st["tracker"].objs.get(tid)
                a, b = st["line"]
                if o and side(o["prev"], a, b) < 0 <= side(o["c"], a, b) and tid in st["decided"]:
                    who, cat = st["decided"][tid]
                    raise_event(name, "enter", who, cat, 1.0, frame, box)
                    
        time.sleep(max(0, interval - (time.time() - t0)))

if __name__ == "__main__":
    run()