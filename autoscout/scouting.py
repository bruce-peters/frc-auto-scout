"""5026's own human match scouting (data/scouting/*.csv, exported from the team's Supabase table
`scouting_submissions`): for one robot in one match, the time of every shot the scout logged.

`scouting_data.shots` holds seconds from match start, one entry per ball (a "+5" tap repeats the
timestamp five times). These are shots taken, not balls scored: misses are not recorded. The app's
timer does not pause between auto and teleop, so a scouted time is video time minus auto start; taps
land ~0.8 s after the launch. Most robots were scouted by two people, so their disagreement measures
how noisy this reference is.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config

SCOUTING = config.DATA / "scouting"


@dataclass
class ScoutRow:
    key: str  # TBA match key, e.g. 2026cc_qm15
    team: int
    alliance: str  # red | blue
    scouter: str
    shots: np.ndarray = field(default_factory=lambda: np.zeros(0))  # match seconds, one per ball
    comments: str = ""

    @property
    def n(self) -> int:
        return len(self.shots)

    @property
    def time_valid(self) -> bool:
        """Timestamps usable for timing (~14 % of rows are post-hoc entries with every tap in auto or
        on one timestamp; their counts may still be fine): <= 50 shots before 20 s, no timestamp
        repeated > 40x, >= 3 distinct timestamps. From the failure-finder audit (cycle 3)."""
        if len(self.shots) == 0:
            return True
        _, c = np.unique(self.shots, return_counts=True)
        return bool((self.shots < 20).sum() <= 50 and c.max() <= 40 and len(c) >= 3)


def _key(sd: dict) -> str | None:
    ev, mt, n = sd.get("event_code"), str(sd.get("match_type") or "").lower(), sd.get("match_number")
    if not ev or not n or ev == "test":
        return None
    if mt.startswith("qual"):
        return f"{ev}_qm{int(n)}"
    return None  # playoff numbering is ambiguous in the export (only quals are used)


def load(paths: list[Path] | None = None) -> list[ScoutRow]:
    import pandas as pd

    rows = []
    for p in paths or sorted(SCOUTING.glob("*.csv")):
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            try:
                sd = json.loads(r["scouting_data"])
            except (TypeError, ValueError):
                continue
            key = _key(sd)
            team = sd.get("team_number") or r.get("team_num")
            role = str(sd.get("role") or r.get("role") or "")
            if key is None or not team or team != team or role[:3] not in ("red", "blu"):
                continue
            rows.append(ScoutRow(key=key, team=int(team), alliance="red" if role.startswith("red") else "blue",
                                 scouter=str(r.get("scouter_id")), shots=np.sort(np.asarray(sd.get("shots") or [], float)),
                                 comments=str(sd.get("comments") or "")))
    return rows


def by_match(rows: list[ScoutRow]) -> dict[str, dict[int, list[ScoutRow]]]:
    out: dict[str, dict[int, list[ScoutRow]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[r.key][r.team].append(r)
    return out
