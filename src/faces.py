import numpy as np
import cv2
from insightface.app import FaceAnalysis

class FaceEngine:
    def __init__(self, det_size=(320, 320)):
        # buffalo_sc is small & fast (SCRFD-500M detector + MobileFaceNet)
        self.app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=det_size)

    def analyse(self, image_bgr):
        faces = self.app.get(image_bgr)
        out = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int)
            out.append({
                "box": (x1, y1, x2, y2),
                "det_score": float(f.det_score),
                "embedding": f.normed_embedding.astype(np.float32), # length 512, unit norm
                "size": min(x2 - x1, y2 - y1)
            })
        return out

def sharpness(crop):
    return cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def good_face(frame, f, min_size=60, min_det=0.6, min_sharp=40):
    x1, y1, x2, y2 = f["box"]
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return False
    return f["size"] > min_size and f["det_score"] >= min_det and sharpness(crop) > min_sharp