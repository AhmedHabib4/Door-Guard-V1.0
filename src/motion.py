import cv2

class MotionDetector:
    def __init__(self, min_area=1200):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)
        self.min_area = min_area

    def detect(self, frame):
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        mask = self.bg.apply(small)
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return any(cv2.contourArea(c) > self.min_area for c in contours)