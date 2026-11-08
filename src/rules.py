RULES = {
    "owner": {"alert": "quiet", "enter": True},
    "family": {"alert": "quiet", "enter": True}, # quiet: short message without sound
    "dustman": {"alert": "normal", "enter": True, "expected_days": [0, 2, 5], "expected_hours": (6, 11)},
    "neighbor": {"alert": "normal", "enter": True},
    "unknown": {"alert": "urgent", "enter": True},
}

def build_message(identity, category, kind, camera, ts):
    if kind == "unknown":
        return f"UNKNOWN person at {camera} ({ts})"
    if kind == "enter":
        return f"{identity} ({category}) ENTERED {camera} ({ts})"
    return f"{identity} ({category}) at the door {camera} ({ts})"

def is_unusual(category, weekday, hour):
    r = RULES.get(category, {})
    if "expected_days" in r:
        lo, hi = r["expected_hours"]
        return weekday not in r["expected_days"] or not (lo < hour < hi)
    return False