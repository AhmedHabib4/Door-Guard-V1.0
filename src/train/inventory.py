import cv2, csv, glob, os

rows = []
for path in sorted(glob.glob("../old_footage/**/*.mp4", recursive=True)):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows.append([path, round(frames/fps/60, 1), round(fps, 1), f"{w}x{h}",
                 os.path.getsize(path) // 1_000_000])
    cap.release()

with open("../data/videos.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["video", "minutes", "fps", "resolution", "MB"])
    wr.writerows(rows)

print(len(rows), "videos,", round(sum(r[1] for r in rows) / 60, 1), "hours")