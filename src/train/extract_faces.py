import cv2, csv, os, numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

MODEL = "buffalo_sc"
SAMPLE_FPS = 3 
MIN_SIZE, MIN_DET, MIN_SHARP = 50, 0.6, 30

app = FaceAnalysis(name=MODEL, providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
os.makedirs("../data/faces", exist_ok=True)
segments = list(csv.DictReader(open("../data/segments.csv")))
meta = csv.writer(open("../data/faces_meta.csv", "w", newline=""))
meta.writerow(["file", "video", "t_s", "det", "size", "sharp"])
n = 0

for seg in segments:
    cap = cv2.VideoCapture(seg["video"])
    vid = os.path.splitext(os.path.basename(seg["video"]))[0].replace(" ", "_")
    t, end = float(seg["start_s"]), float(seg["end_s"])
    
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        t += 1.0 / SAMPLE_FPS
        if not ok:
            continue
            
        for f in app.get(frame):
            x1, y1, x2, y2 = f.bbox.astype(int)
            size = min(x2 - x1, y2 - y1)
            if size < MIN_SIZE or f.det_score < MIN_DET:
                continue
                
            aligned = face_align.norm_crop(frame, f.kps, image_size=112)
            sharp = cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            if sharp < MIN_SHARP:
                continue
                
            fn = f"../data/faces/{vid}_{int(t * 1000):08d}_{n}.jpg"
            cv2.imwrite(fn, aligned)
            meta.writerow([fn, vid, round(t, 2), round(float(f.det_score), 3), size, round(sharp, 1)])
            n += 1
    cap.release()
    print(seg["video"], "total crops so far:", n)