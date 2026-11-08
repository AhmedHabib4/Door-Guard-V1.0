import os, cv2, numpy as np
from insightface.model_zoo import get_model

REC = os.path.expanduser("~/.insightface/models/buffalo_sc/w600k_mbf.onnx")
rec = get_model(REC, providers=["CPUExecutionProvider"])
rec.prepare(ctx_id=-1)

def embed_batch(imgs):
    f = rec.get_feat(imgs)
    return (f / np.linalg.norm(f, axis=1, keepdims=True)).astype(np.float32)

def embed_files(files, batch=64):
    out = []
    for i in range(0, len(files), batch):
        imgs = [cv2.imread(f) for f in files[i:i + batch]]
        out.append(embed_batch(imgs))
    return np.concatenate(out)

if __name__ == "__main__":
    import csv
    files = [r["file"] for r in csv.DictReader(open("data/faces_meta.csv"))]
    np.save("data/emb_crops.npy", embed_files(files))
    print("saved", len(files), "embeddings")