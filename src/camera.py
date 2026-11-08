import cv2, threading, time

class CameraStream:
    def __init__(self, name, url):
        self.name, self.url = name, url
        self.frame = None
        self.last_ok = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open(self):
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _loop(self):
        cap = self._open()
        while self.running:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                time.sleep(3)
                cap = self._open()
                continue
            with self.lock:
                self.frame = frame
                self.last_ok = time.time()
        cap.release()

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def is_alive(self):
        return time.time() - self.last_ok < 10

    def stop(self):
        self.running = False