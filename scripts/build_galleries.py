"""Per-(event, team) appearance galleries from confidently identified segments.

A segment contributes to its assigned team's gallery when its bumper-OCR evidence is strong
(>= 2) and points at that team (>= 80 % of its OCR mass). Contributions are kept per match so
that a match's own segments can be excluded when the gallery is used on that match.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, pipeline  # noqa: E402

OUT = config.MODELS / "galleries.pkl"


def main():
    contrib = {}  # (event, team) -> {match: (sum vec, weight)}
    for f in sorted(config.OUTPUT.glob("*/scan.pkl")):
        key = f.parent.name
        with open(f, "rb") as fh:
            md = pickle.load(fh)
        res = pipeline.analyse(md, pipeline.Options())
        team_of = {}
        for al in ("red", "blue"):
            for sid, s in res["identity"][al].items():
                team_of[int(sid)] = s["team"]
        ev_key = md.info.event_key
        n = 0
        for sid, (vec, ocr_ev, teams) in md.seg_app.items():
            team = team_of.get(sid)
            if team is None or ocr_ev.sum() < 2:
                continue
            if ocr_ev[teams.index(team)] < 0.8 * ocr_ev.sum():
                continue
            v = vec / (np.linalg.norm(vec) + 1e-9)
            w = float(ocr_ev.sum())
            d = contrib.setdefault((ev_key, team), {})
            sv, sw = d.get(key, (np.zeros_like(v), 0.0))
            d[key] = (sv + w * v, sw + w)
            n += 1
        print(f"{key}: {n} confident segments", flush=True)
    with open(OUT, "wb") as fh:
        pickle.dump(contrib, fh)
    print(f"galleries for {len(contrib)} (event, team) pairs -> {OUT}")


def galleries_for(match_key: str, event_key: str) -> dict:
    """Leave-this-match-out galleries for one match."""
    if not OUT.exists():
        return {}
    with open(OUT, "rb") as fh:
        contrib = pickle.load(fh)
    out = {}
    for (ev, team), d in contrib.items():
        if ev != event_key:
            continue
        sv, sw = None, 0.0
        for m, (v, w) in d.items():
            if m == match_key:
                continue
            sv = v.copy() if sv is None else sv + v
            sw += w
        if sv is not None and sw > 0:
            out[team] = sv / (np.linalg.norm(sv) + 1e-9)
    return out


if __name__ == "__main__":
    main()
