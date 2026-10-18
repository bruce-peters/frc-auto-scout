"""Fuel OPR: least-squares per-team fuel contribution from alliance totals over an event.

Evaluation reference (a per-team number that does not come from video): if the video attribution is right, a
team's average attributed fuel over an event should track its fuel OPR. The learned credit model also uses it as
a prior, computed only from the matches played before the one being analysed (`before`, pipeline.NO_HINDSIGHT).
"""
from __future__ import annotations

import numpy as np

from . import tba


LEVEL_ORDER = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}


def match_order(m: tba.MatchInfo) -> tuple:
    """Schedule order within an event: quals by number, then playoffs by level, set and match."""
    return (LEVEL_ORDER.get(m.comp_level, 9), m.set_number, m.match_number)


def fuel_opr(event_key: str, comp_levels=("qm",), ridge: float = 0.5, before: str | None = None) -> dict[int, float]:
    """Fuel OPR over the event's `comp_levels` matches; with `before` (a match key) only the matches played
    before it, as a live system would have it ({} before the first scored match)."""
    ms = [m for m in tba.event_matches(event_key) if m.comp_level in comp_levels and m.breakdown]
    if before is not None:
        cut = match_order(tba.match(before))
        ms = [m for m in ms if match_order(m) < cut]
    teams = sorted({t for m in ms for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    rows, y = [], []
    for m in ms:
        for al in ("red", "blue"):
            hub = m.hub(al)
            if not hub:
                continue
            r = np.zeros(len(teams))
            for t in m.teams(al):
                r[idx[t]] = 1
            rows.append(r)
            y.append(hub.get("totalCount", 0))
    if not rows:
        return {}
    A, y = np.array(rows), np.array(y, dtype=float)
    # ridge towards the mean contribution keeps teams with few matches sane
    mu = y.mean() / 3
    x = np.linalg.solve(A.T @ A + ridge * np.eye(len(teams)), A.T @ (y - A @ np.full(len(teams), mu)))
    x += mu
    return {t: float(max(0.0, x[idx[t]])) for t in teams}
