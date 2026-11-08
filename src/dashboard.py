from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
import database as db

app = FastAPI()
con = db.connect()

@app.get("/api/events")
def events(limit: int = 50):
    rows = con.execute("SELECT ts, camera, kind, identity, category, score, snapshot FROM events "
                       "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(zip(["ts", "camera", "kind", "identity", "category", "score", "snapshot"], r)) for r in rows]

@app.get("/snap")
def snap(path: str):
    if not path.startswith("../snapshots/") or ".." in path.replace("../snapshots/", ""):
        return {"error": "forbidden"}
    return FileResponse(path)

@app.get("/", response_class=HTMLResponse)
def home():
    rows = events(30)
    tr = "".join(f"<tr><td>{r['ts']}</td><td>{r['camera']}</td><td>{r['kind']}</td>"
                 f"<td>{r['identity']}</td>"
                 f"<td><a href='/snap?path={r['snapshot']}'>photo</a></td></tr>" for r in rows)
    head = "<tr><th>Time</th><th>Camera</th><th>Event</th><th>Who</th><th>Image</th></tr>"
    return f"<h2>Door events</h2><table border=1 cellpadding=6>{head}{tr}</table>"