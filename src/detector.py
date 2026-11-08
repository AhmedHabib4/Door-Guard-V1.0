from ultralytics import YOLO

PERSON, VEHICLES = 0, [2, 3, 5, 7] # car, motorcycle, bus, truck

class ObjectDetector:
    def __init__(self, size=480, conf=0.4):
        self.model = YOLO("yolov8n.pt") # downloads on first run
        self.size, self.conf = size, conf

    def detect(self, frame):
        res = self.model(frame, imgsz=self.size, conf=self.conf, classes=[0, 2, 3, 5, 7], verbose=False)[0]
        out = []
        for b in res.boxes:
            cls = int(b.cls[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            out.append({"cls": cls, "conf": float(b.conf[0]), "box": (x1, y1, x2, y2)})
        return out

    def persons(self, dets):
        return [d for d in dets if d["cls"] == PERSON]

    def vehicles(self, dets):
        return [d for d in dets if d["cls"] in VEHICLES]