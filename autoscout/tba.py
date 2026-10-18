"""The Blue Alliance API v3 client with an on-disk JSON cache."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .config import CACHE, TBA_API_KEY

BASE = "https://www.thebluealliance.com/api/v3"
_CACHE_DIR = CACHE / "tba"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get(path: str, max_age_s: float = 7 * 24 * 3600) -> object:
    """GET a TBA endpoint, caching the JSON response on disk."""
    fn = _CACHE_DIR / (path.strip("/").replace("/", "__") + ".json")
    if fn.exists() and time.time() - fn.stat().st_mtime < max_age_s:
        return json.loads(fn.read_text())
    if not TBA_API_KEY:
        raise RuntimeError("TBA_API_KEY missing from .env")
    r = requests.get(BASE + path, headers={"X-TBA-Auth-Key": TBA_API_KEY}, timeout=30)
    r.raise_for_status()
    data = r.json()
    fn.write_text(json.dumps(data))
    return data


@dataclass
class MatchInfo:
    key: str
    event_key: str
    comp_level: str
    match_number: int
    set_number: int
    red: list[int]  # station order 1..3
    blue: list[int]
    youtube: list[str]
    breakdown: dict | None
    red_score: int
    blue_score: int

    def alliance_of(self, team: int) -> str | None:
        if team in self.red:
            return "red"
        if team in self.blue:
            return "blue"
        return None

    def teams(self, alliance: str) -> list[int]:
        return self.red if alliance == "red" else self.blue

    def hub(self, alliance: str) -> dict:
        return (self.breakdown or {}).get(alliance, {}).get("hubScore", {})


def _team_num(k: str) -> int:
    return int(k.replace("frc", "").rstrip("BCDE"))


def parse_match(m: dict) -> MatchInfo:
    return MatchInfo(
        key=m["key"],
        event_key=m["event_key"],
        comp_level=m["comp_level"],
        match_number=m["match_number"],
        set_number=m["set_number"],
        red=[_team_num(k) for k in m["alliances"]["red"]["team_keys"]],
        blue=[_team_num(k) for k in m["alliances"]["blue"]["team_keys"]],
        youtube=[v["key"] for v in m.get("videos", []) if v.get("type") == "youtube"],
        breakdown=m.get("score_breakdown"),
        red_score=m["alliances"]["red"]["score"],
        blue_score=m["alliances"]["blue"]["score"],
    )


def match(key: str) -> MatchInfo:
    return parse_match(get(f"/match/{key}"))


def event_matches(event_key: str) -> list[MatchInfo]:
    ms = [parse_match(m) for m in get(f"/event/{event_key}/matches")]
    order = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
    ms.sort(key=lambda m: (order.get(m.comp_level, 9), m.set_number, m.match_number))
    return ms


def team_events(team: int, year: int = 2026) -> list[dict]:
    return get(f"/team/frc{team}/events/{year}/simple")


def team_event_matches(team: int, event_key: str) -> list[MatchInfo]:
    return [m for m in event_matches(event_key) if team in m.red + m.blue]


def tower_points(mi: MatchInfo, alliance: str) -> dict[int, dict]:
    """Per-team tower points from the official per-robot breakdown (station order)."""
    from .config import TOWER_POINTS

    bd = (mi.breakdown or {}).get(alliance, {})
    out = {}
    for i, team in enumerate(mi.teams(alliance), start=1):
        a = bd.get(f"autoTowerRobot{i}", "None")
        e = bd.get(f"endGameTowerRobot{i}", "None")
        out[team] = {
            "auto_tower": a,
            "endgame_tower": e,
            "auto_tower_points": TOWER_POINTS["auto"].get(a, 0),
            "endgame_tower_points": TOWER_POINTS["endgame"].get(e, 0),
        }
    return out


def inactive_periods(mi: MatchInfo) -> dict[str, set[str]]:
    """Which teleop shifts each alliance's hub is inactive (game manual 6.4.1).

    The alliance with more AUTO fuel is inactive in SHIFT 1 and 3; the other in SHIFT 2 and 4.
    TBA records 0 fuel for inactive shifts, which resolves auto ties (broken randomly by FMS).
    """
    r, b = mi.hub("red"), mi.hub("blue")
    first = None
    if r and b:
        if r.get("autoCount", 0) != b.get("autoCount", 0):
            first = "red" if r.get("autoCount", 0) > b.get("autoCount", 0) else "blue"
        else:
            r13 = r.get("shift1Count", 0) + r.get("shift3Count", 0)
            b13 = b.get("shift1Count", 0) + b.get("shift3Count", 0)
            first = "red" if r13 < b13 else "blue"
    if first is None:
        return {"red": set(), "blue": set()}
    other = "blue" if first == "red" else "red"
    return {first: {"shift1", "shift3"}, other: {"shift2", "shift4"}}
