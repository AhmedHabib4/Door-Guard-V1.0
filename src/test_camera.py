import cv2, yaml, time
from camera import CameraStream

cfg = yaml.safe_load(open("../config.yaml"))
cams = [CameraStream(c["name"], c["rtsp_sub"]) for c in cfg["cameras"]]
time.sleep(3)

while True:
    for c in cams:
        f = c.read()
        if f is not None:
            cv2.imshow(c.name, f)
    if cv2.waitKey(30) == 27:
        break