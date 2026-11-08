import glob, os, numpy as np
from embed import embed_files

for folder in sorted(glob.glob("data/labelled/*")):
    files = glob.glob(folder + "/*.jpg")
    if len(files) < 5:
        continue
    E = embed_files(files)
    center = E.mean(0)
    center /= np.linalg.norm(center)
    sims = E @ center
    bad = [os.path.basename(files[i]) for i in np.where(sims < 0.35)[0]]
    name = os.path.basename(folder)
    
    print(f"{name:28s} n={len(files):4d} mean_sim={sims.mean():.2f} suspicious={len(bad)}")
    for b in bad[:10]:
        print("  check", b)