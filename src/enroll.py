import os, cv2, sys
from faces import FaceEngine
import database as db

engine = FaceEngine(det_size=(640, 640))
con = db.connect()
root = "../dataset"

for folder in sorted(os.listdir(root)):
    if "_" not in folder:
        continue
    category, name = folder.split("_", 1)
    pid = db.add_person(con, name.replace("_", " "), category)
    added = 0
    for fn in sorted(os.listdir(os.path.join(root, folder))):
        img = cv2.imread(os.path.join(root, folder, fn))
        if img is None:
            continue
        faces = engine.analyse(img)
        if len(faces) != 1:
            print(f"skip {folder}/{fn}: {len(faces)} faces")
            continue
        db.add_embedding(con, pid, faces[0]["embedding"], fn)
        added += 1
    print(f"{name}: {added} embeddings")