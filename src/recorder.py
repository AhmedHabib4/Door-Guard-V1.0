import subprocess, time, yaml, os

cfg = yaml.safe_load(open("../config.yaml"))
procs = {}

def start(cam):
    os.makedirs(f"../recordings/{cam['name']}", exist_ok=True)
    cmd = ["ffmpeg", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", cam["rtsp_main"],
           "-c", "copy", "-f", "segment", "-segment_time", "600", "-reset_timestamps", "1",
           "-strftime", "1", f"../recordings/{cam['name']}/%Y-%m-%d_%H%M-%S.mp4"]
    return subprocess.Popen(cmd)

while True:
    for cam in cfg["cameras"]:
        p = procs.get(cam["name"])
        if p is None or p.poll() is not None:
            procs[cam["name"]] = start(cam)
    time.sleep(5)