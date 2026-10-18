"""Hand labels of who scored: per-alliance "scoring bursts" and their labelled shooters.

The alliance FUEL counter (read from the broadcast overlay) is exact, so the only thing a
person needs to label to get per-team ground truth is *who launched* the balls behind each
jump of the counter. Consecutive counter increments are grouped into bursts (gap <= 1.5 s,
at most 5 s long, since the shooter can change); the labeller watches the few seconds before
each burst and marks the robot(s) that launched those balls.

Labels live in data/labels/<match>__<labeller>.json, one file per person, so several people can
label the same match (and their agreement can be measured). Each label stores its own time
window and ball count, so it stays valid even if the burst list is recomputed later.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from . import config

LABELS = config.DATA / "labels"
GAP_S = 1.5  # a pause longer than this starts a new burst
MAX_LEN_S = 5.0  # long bursts are split; the shooter can change
LEAD_S = 4.0  # play this long before the first increment (launch -> count is ~1-5 s)
TAIL_S = 0.5
NAME_RE = re.compile(r"^[A-Za-z0-9 _.-]{1,40}$")


def _to_mt(tl: dict):
    a0, ts = tl.get("auto_start") or 6.13, tl.get("teleop_start")

    def f(vt: float) -> float:
        return (20 + vt - ts) if ts and vt >= ts else vt - a0
    return f


def bursts(res: dict) -> list[dict]:
    """Scoring bursts of both alliances, in time order."""
    tl = res.get("timeline") or {}
    to_mt = _to_mt(tl)
    out = []
    for al in ("red", "blue"):
        inc, prev = [], 0
        for t, c in (tl.get("counts") or {}).get(al) or []:
            if c > prev:
                inc.append((float(t), int(c - prev)))
            prev = max(prev, c)
        groups: list[list] = []
        for t, n in inc:
            if groups and t - groups[-1][-1][0] <= GAP_S and t - groups[-1][0][0] <= MAX_LEN_S:
                groups[-1].append((t, n))
            else:
                groups.append([(t, n)])
        for g in groups:
            t0, t1 = g[0][0], g[-1][0]
            mt0 = to_mt(t0 - 2.0)  # the counter lags the ball entering the hub by ~2 s
            if mt0 > 165:  # after the match
                continue
            out.append({"id": f"{al}-{int(round(t0 * 100))}", "alliance": al, "t0": round(t0, 3),
                        "t1": round(t1, 3), "balls": int(sum(n for _, n in g)), "mt": round(mt0, 1),
                        "play0": round(max(0.0, t0 - LEAD_S), 3), "play1": round(t1 - TAIL_S, 3)})
    return sorted(out, key=lambda b: b["t0"])


def _path(match_key: str, labeller: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", labeller.strip())
    return LABELS / f"{match_key}__{safe}.json"


def load(match_key: str, labeller: str) -> dict:
    p = _path(match_key, labeller)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"match": match_key, "labeller": labeller, "labels": {}}


def save(match_key: str, labeller: str, labels: dict) -> dict:
    """labels: {burst id: {alliance, t0, t1, balls, shooters: [team...], status, note}}."""
    if not NAME_RE.match(labeller or ""):
        raise ValueError("labeller name: 1-40 letters, digits, spaces, _ . -")
    LABELS.mkdir(parents=True, exist_ok=True)
    clean = {}
    for bid, lab in (labels or {}).items():
        status = lab.get("status")
        if status not in ("labeled", "unsure", "other", "skip"):
            continue
        clean[str(bid)] = {
            "alliance": lab.get("alliance"), "t0": float(lab["t0"]), "t1": float(lab["t1"]),
            "balls": int(lab.get("balls", 0)), "status": status,
            "shooters": [int(s) for s in lab.get("shooters", []) if str(s).isdigit()],
            "note": str(lab.get("note", ""))[:200], "at": lab.get("at") or time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    doc = {"match": match_key, "labeller": labeller, "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "labels": clean}
    _path(match_key, labeller).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def all_label_files(match_key: str | None = None) -> list[Path]:
    pat = f"{match_key}__*.json" if match_key else "*__*.json"
    return sorted(LABELS.glob(pat)) if LABELS.exists() else []


def progress(match_key: str) -> dict:
    """{labeller: number of bursts labelled} for one match."""
    out = {}
    for p in all_label_files(match_key):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            out[d.get("labeller", p.stem.split("__", 1)[-1])] = len(d.get("labels", {}))
        except Exception:
            continue
    return out
