import cv2, csv, glob, os
import sys
sys.path.append("../src")
from detector import ObjectDetector

det = ObjectDetector(size=640, conf=0.4)
os.makedirs("../data/vehicles/raw", exist_ok=True)
meta = csv.writer(open("../data/vehicles_meta.csv", "w", newline=""))
meta.writerow(["file", "video", "t_s"])
n = 0

for path in sorted(glob.glob("../old_footage/**/*.mp4", recursive=True)):
    vid = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = int(fps) # 1 frame per second
    i = 0
    
    while cap.grab():
        if i % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                for d in det.vehicles(det.detect(frame)):
                    x1, y1, x2, y2 = d["box"]
                    if (x2 - x1) < 120:
                        continue # ignore tiny, far vehicles
                    fn = f"../data/vehicles/raw/{vid}_{i}_{n}.jpg"
                    cv2.imwrite(fn, frame[max(0, y1):y2, max(0, x1):x2])
                    meta.writerow([fn, vid, round(i / fps, 1)])
                    n += 1
        i += 1
    cap.release()