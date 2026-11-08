import os, time

KEEP_DAYS = 10
now = time.time()

for root, dirs, files in os.walk("../recordings"):
    for f in files:
        path = os.path.join(root, f)
        if now - os.path.getmtime(path) > KEEP_DAYS * 86400:
            os.remove(path)

for root, dirs, files in os.walk("../snapshots"):
    for f in files:
        path = os.path.join(root, f)
        if now - os.path.getmtime(path) > 30 * 86400: # snapshots kept 30 days
            os.remove(path)