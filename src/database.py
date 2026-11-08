import sqlite3, numpy as np, datetime

def connect(path="../db/door.sqlite"):
    con = sqlite3.connect(path, check_same_thread=False)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS people(
            id INTEGER PRIMARY KEY, name TEXT UNIQUE, category TEXT);
        CREATE TABLE IF NOT EXISTS face_embeddings(
            id INTEGER PRIMARY KEY, person_id INTEGER, embedding BLOB, source TEXT);
        CREATE TABLE IF NOT EXISTS vehicles(
            id INTEGER PRIMARY KEY, plate TEXT UNIQUE, owner TEXT, category TEXT);
        CREATE TABLE IF NOT EXISTS events(
            id INTEGER PRIMARY KEY, ts TEXT, camera TEXT, kind TEXT,
            identity TEXT, category TEXT, score REAL, snapshot TEXT);
    """)
    return con

def add_person(con, name, category):
    con.execute("INSERT OR IGNORE INTO people (name, category) VALUES(?,?)", (name, category))
    con.commit()
    return con.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()[0]

def add_embedding(con, person_id, emb, source):
    con.execute("INSERT INTO face_embeddings (person_id, embedding, source) VALUES(?,?,?)",
                (person_id, emb.astype(np.float32).tobytes(), source))
    con.commit()

def log_event(con, camera, kind, identity, category, score, snapshot):
    con.execute("INSERT INTO events (ts, camera, kind, identity, category, score, snapshot) "
                "VALUES(?,?,?,?,?,?,?)",
                (datetime.datetime.now().isoformat(timespec="seconds"), camera, kind,
                 identity, category, float(score), snapshot))
    con.commit()