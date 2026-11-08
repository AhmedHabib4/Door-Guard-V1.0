import cv2, csv, glob, sys
sys.path.append("../src")
from motion import MotionDetector
from detector import ObjectDetector

det = ObjectDetector(size=416)
out = csv.writer(open("../data/segments.csv", "w", newline=""))
out.writerow(["video", "start_s", "end_s"])

for path in sorted(glob.glob("../old_footage/**/*.mp4", recursive=True)):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(fps / 2)) # analyse 2 frames per second
    md = MotionDetector()
    i, start, last = 0, None, None

    while cap.grab(): # grab() skips decoding colour conversion: faster
        if i % step == 0:
            ok, frame = cap.retrieve()
            t = i / fps
            has_person = False
            if ok and md.detect(frame):
                has_person = len(det.persons(det.detect(frame))) > 0
            
            if has_person:
                start = t if start is None else start
                last = t
            elif start is not None and t - last > 5: # 5 s without a person closes the segment
                out.writerow([path, max(0, start - 2), last + 2])
                start = None
        i += 1
    
    if start is not None:
        out.writerow([path, max(0, start - 2), last + 2])
    cap.release()
    print("done", path)