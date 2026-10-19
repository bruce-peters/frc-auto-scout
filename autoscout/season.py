"""Season-to-date fuel prior: each team's fuel scoring rate from its earlier events of the season.

The learned credit model (attribution.ATTR_MODEL) pulls each team's share of its alliance's fuel toward its
fuel-OPR share. With no hindsight (pipeline.NO_HINDSIGHT) that OPR comes only from the event's quals played
before the match, so the first matches of an event get no prior and the next ones a noisy one. Every team's
earlier events of the season are public before an event starts; this module turns them into a per-team rate
and blends it with the event's own matches as they are played.

* season_rate(team, event_key): fuel OPR (full quals, from TBA hub totals) at each earlier event of the same
  season that ended before event_key started, averaged with weights (quals played) x
  0.5^(days before the event / HALF_LIFE_DAYS). NORM divides each event's OPR by its scoring level first (team +
  event effects fit); it predicted worse on both tuning sets, so it is off.
* prior_oprs(event_key, before): {team: rate} for the event's teams, the drop-in for opr.fuel_opr(before=).
  Fuel OPR of the quals played before `before`, ridge-shrunk toward each team's season rate (scaled to the
  event's level) with a strength of SEASON_WEIGHT alliance-matches per earlier match.
* season_prior(team, event_key, before_match): one team's entry of prior_oprs.

SOURCE picks the season rate: "opr" (above, the default), "epa" (Statbotics' fuel EPA update re-run on TBA hub
totals, as of the event start: season_epa) or "statbotics" (Statbotics' own total_fuel EPA after the team's
last earlier event, from its API). TBA responses are cached on disk by tba.get (data/cache/tba), Statbotics
responses in data/cache/statbotics; Statbotics is also the fallback for a team with no usable TBA history.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache

import numpy as np
import requests

from . import opr as oprmod
from . import tba
from .config import CACHE

# tuned on the 8 Champs divisions and 14 summer offseasons, not on Chezy (data/exp_season/RESULTS_section.md)
HALF_LIFE_DAYS = 60.0  # recency: an event this many days older counts half as much
SEASON_WEIGHT = 0.5  # ridge strength per earlier-event qual, in alliance-match observations of this event
NORM = False  # divide each event's OPR by its fitted scoring level (event_levels) before averaging
SOURCE = "opr"  # season rate: "opr" | "epa" | "statbotics" (module docstring)
MIN_QUALS = 8  # earlier events with fewer scored quals (with hub totals) are skipped
EVENT_RIDGE = 2.0  # shrinks weakly connected events' level toward the average (team-event equivalents)
NO_PRIOR_RIDGE = 0.5  # opr.fuel_opr's ridge, for a team with no season rate (toward the event mean)
SKIP_TYPES = {100}  # preseason

STATBOTICS = "https://api.statbotics.io/v3"
_SB_DIR = CACHE / "statbotics"
_SB_FAILED: set[str] = set()


# ----------------------------------------------------------------------------- TBA history
@lru_cache(maxsize=None)
def event_simple(event_key: str) -> dict:
    return tba.get(f"/event/{event_key}/simple")


def _day(s: str) -> date:
    return date.fromisoformat(s)


def prior_events(team: int, event_key: str) -> list[dict]:
    """The team's events of the same season that ended before event_key started (preseason excluded)."""
    ev = event_simple(event_key)
    start = _day(ev["start_date"])
    out = []
    for e in tba.team_events(team, int(ev["year"])):
        if e["key"] == event_key or e.get("event_type") in SKIP_TYPES:
            continue
        if _day(e["end_date"]) < start:
            out.append(e)
    return sorted(out, key=lambda e: e["end_date"])


@lru_cache(maxsize=None)
def event_oprs(event_key: str) -> tuple[dict[int, float], dict[int, int]]:
    """Full-quals fuel OPR of a finished event and quals played per team ({} if under MIN_QUALS scored)."""
    try:
        ms = [m for m in tba.event_matches(event_key) if m.comp_level == "qm" and m.hub("red") and m.hub("blue")]
    except requests.HTTPError:
        return {}, {}
    if len(ms) < MIN_QUALS:
        return {}, {}
    n: dict[int, int] = {}
    for m in ms:
        for t in m.red + m.blue:
            n[t] = n.get(t, 0) + 1
    return oprmod.fuel_opr(event_key), n


@lru_cache(maxsize=None)
def event_teams(event_key: str) -> tuple[int, ...]:
    return tuple(sorted({tba._team_num(k) for k in tba.get(f"/event/{event_key}/teams/keys")}))


@lru_cache(maxsize=None)
def event_levels(event_key: str) -> dict[str, float]:
    """Scoring level (multiplier) of every earlier event of event_key's teams, from log fuel OPR = team + event
    effects fitted over all teams at those events (weights min(quals, 12) / 12; event effects ridge-shrunk
    toward 0 by EVENT_RIDGE). Geometric weighted mean 1 over the events."""
    keys = sorted({e["key"] for t in event_teams(event_key) for e in prior_events(t, event_key)})
    rows = []  # (team, event index, log opr, weight)
    for j, k in enumerate(keys):
        o, n = event_oprs(k)
        if not o:
            continue
        mean = np.mean(list(o.values()))
        for t, x in o.items():
            rows.append((t, j, np.log(max(x, 0.1 * mean)), min(n.get(t, 0), 12) / 12))
    by_team: dict[int, list] = {}
    for t, j, z, w in rows:
        if w > 0:
            by_team.setdefault(t, []).append((j, z, w))
    # eliminate the team effects: for each team, residuals about its weighted mean constrain the event effects
    E = len(keys)
    M = EVENT_RIDGE * np.eye(E)
    b = np.zeros(E)
    for obs in by_team.values():
        if len(obs) < 2:
            continue
        j = np.array([o[0] for o in obs])
        z = np.array([o[1] for o in obs])
        w = np.array([o[2] for o in obs])
        W = w.sum()
        Mt = np.diag(w) - np.outer(w, w) / W
        np.add.at(M, (j[:, None], j[None, :]), Mt)
        np.add.at(b, j, w * z - w * (w @ z) / W)
    mu = np.linalg.solve(M, b)
    used = np.zeros(E, bool)
    for _, j, _, _ in rows:
        used[j] = True
    if used.any():
        mu -= mu[used].mean()
    return {k: float(np.exp(mu[j])) for j, k in enumerate(keys) if used[j]}


@dataclass
class SeasonRate:
    rate: float  # fuel per match, in level-normalised units (raw fuel OPR units if NORM is False)
    n_eff: float  # recency-discounted quals behind it
    events: list = field(default_factory=list)  # (event key, fuel OPR, level, quals, weight)
    source: str = "tba"


def season_rate(team: int, event_key: str, half_life: float = None, norm: bool = None,
                source: str = None) -> SeasonRate | None:
    """The team's fuel scoring rate from its earlier events of the season (None without usable history)."""
    half_life = HALF_LIFE_DAYS if half_life is None else half_life
    norm = NORM if norm is None else norm
    source = SOURCE if source is None else source
    if source == "epa":
        return epa_rate(team, event_key, half_life)
    if source == "statbotics":
        return statbotics_rate(team, event_key, half_life)
    start = _day(event_simple(event_key)["start_date"])
    levels = event_levels(event_key) if norm else {}
    num = den = 0.0
    evs = []
    for e in prior_events(team, event_key):
        o, n = event_oprs(e["key"])
        if team not in o or not n.get(team):
            continue
        age = (start - _day(e["end_date"])).days
        w = n[team] * (0.5 ** (age / half_life) if half_life and np.isfinite(half_life) else 1.0)
        lv = levels.get(e["key"], 1.0) if norm else 1.0
        num += w * o[team] / lv
        den += w
        evs.append((e["key"], round(o[team], 2), round(lv, 3), n[team], round(w, 3)))
    if den > 0:
        return SeasonRate(num / den, den, evs)
    return statbotics_rate(team, event_key, half_life)


def _quals_eff(team: int, event_key: str, half_life: float) -> float:
    """The team's quals at its earlier events, recency-weighted as in season_rate (the blend's prior strength)."""
    start = _day(event_simple(event_key)["start_date"])
    tot = 0.0
    for e in prior_events(team, event_key):
        n = event_oprs(e["key"])[1].get(team, 0)
        tot += n * (0.5 ** ((start - _day(e["end_date"])).days / half_life) if np.isfinite(half_life) else 1.0)
    return tot


# ----------------------------------------------------------------------------- Statbotics-style EPA
EPA_ELIM_WEIGHT = 1 / 3  # Statbotics models/epa/constants.py ELIM_WEIGHT
EPA_INIT_MATCHES = 200  # every team starts at a third of the mean alliance fuel of the season's first matches


def _epa_percent(n_quals: int) -> float:
    """Statbotics' EPA learning rate (models/epa/main.py percent_func, years >= 2016): 1/3 of the alliance error
    share for a team's first 6 quals, falling to 0.2 by its 12th."""
    return 2 / 3 * min(0.5, max(0.3, 0.5 - 0.2 / 6 * (n_quals - 6)))


@lru_cache(maxsize=None)
def season_epa(year: int) -> dict:
    """Statbotics' fuel EPA (models/epa/main.py) re-run on TBA hub totals over every scored match of the season,
    in match-time order. Each alliance's error (its fuel minus the sum of its teams' EPAs) is split evenly over
    its 3 teams and each EPA moves by _epa_percent of its share; playoffs count EPA_ELIM_WEIGHT and do not
    advance the qual count; the margin (opponent) term is 0, as Statbotics has it for 2026. Differences from
    Statbotics: no carry-over from the previous season (Statbotics starts each team from its earlier years'
    EPA), and the start is the mean of the season's first EPA_INIT_MATCHES matches, not of the whole season.
    {team: [(match time, EPA after the match, event key, event end date, qual?)]}; B/C teams keep their key."""
    events = {e["key"]: e for e in tba.get(f"/events/{year}/simple")
              if e.get("event_type") not in SKIP_TYPES and _day(e["start_date"]) <= date.today()}
    rows = []
    for k, e in events.items():
        try:
            ms = tba.get(f"/event/{k}/matches")
        except requests.HTTPError:
            continue
        noon = time.mktime(_day(e["end_date"]).timetuple()) + 12 * 3600
        for m in ms:
            bd = m.get("score_breakdown") or {}
            ys = [((bd.get(a) or {}).get("hubScore") or {}).get("totalCount") for a in ("red", "blue")]
            if None in ys:
                continue
            t = m.get("actual_time") or m.get("time") or m.get("predicted_time") or noon
            order = (oprmod.LEVEL_ORDER.get(m["comp_level"], 9), m["set_number"], m["match_number"])
            rows.append((t, k, order, m, ys))
    rows.sort(key=lambda x: x[:3])
    if not rows:
        return {}
    init = float(np.mean([y for r in rows[:EPA_INIT_MATCHES] for y in r[4]])) / 3

    def tid(key):
        return int(key[3:]) if key[3:].isdigit() else key

    epa: dict = {}
    nq: dict = {}
    hist: dict = {}
    for t, k, _, m, ys in rows:
        qual = m["comp_level"] == "qm"
        w = 1.0 if qual else EPA_ELIM_WEIGHT
        als = [[tid(x) for x in m["alliances"][a]["team_keys"]] for a in ("red", "blue")]
        errs = [y - sum(epa.get(x, init) for x in al) for al, y in zip(als, ys)]  # both before either update
        end = _day(events[k]["end_date"])
        for al, err in zip(als, errs):
            for x in al:
                epa[x] = epa.get(x, init) + w * _epa_percent(nq.get(x, 0)) * err / len(al)
                if qual:
                    nq[x] = nq.get(x, 0) + 1
                hist.setdefault(x, []).append((t, epa[x], k, end, qual))
    return hist


def epa_rate(team: int, event_key: str, half_life: float = None) -> SeasonRate | None:
    """The team's fuel EPA (season_epa) after its last match at an event that ended before event_key started."""
    half_life = HALF_LIFE_DAYS if half_life is None else half_life
    ev = event_simple(event_key)
    start = _day(ev["start_date"])
    prior = [h for h in season_epa(int(ev["year"])).get(team, []) if h[3] < start and h[2] != event_key]
    if not prior:
        return statbotics_rate(team, event_key, half_life)
    evs = sorted({(h[2], h[3]) for h in prior}, key=lambda x: x[1])
    return SeasonRate(prior[-1][1], _quals_eff(team, event_key, half_life), [k for k, _ in evs], "epa")


# ----------------------------------------------------------------------------- Statbotics fallback
def _sb_get(path: str, max_age_s: float = 7 * 24 * 3600) -> object | None:
    """GET a Statbotics v3 endpoint with an on-disk cache; None on any failure (the API is often down)."""
    _SB_DIR.mkdir(parents=True, exist_ok=True)
    fn = _SB_DIR / (path.strip("/").replace("/", "__").replace("?", "_").replace("&", "_").replace("=", "-") + ".json")
    if fn.exists() and time.time() - fn.stat().st_mtime < max_age_s:
        return json.loads(fn.read_text())
    if path in _SB_FAILED:
        return None
    try:
        r = requests.get(STATBOTICS + path, timeout=20)
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError):
        _SB_FAILED.add(path)  # not retried in this process
        return None
    fn.write_text(json.dumps(data))
    return data


def _sb_fuel(epa: dict) -> float | None:
    """total_fuel of a Statbotics 2026 EPA breakdown (backend/src/breakdown.py: auto + transition + first and
    second shift + endgame fuel), as a number or a {"mean": ...} entry."""
    x = ((epa or {}).get("breakdown") or {}).get("total_fuel")
    if isinstance(x, dict):
        x = x.get("mean")
    return float(x) if isinstance(x, (int, float)) else None


def statbotics_rate(team: int, event_key: str, half_life: float = None) -> SeasonRate | None:
    """Statbotics' total_fuel EPA at the end of the team's last earlier event (team_event endpoint), i.e. as of
    the event start; None if the API has nothing (it is often down)."""
    half_life = HALF_LIFE_DAYS if half_life is None else half_life
    if not prior_events(team, event_key):
        return None
    year = int(event_simple(event_key)["year"])
    by_event = {d.get("event"): d for d in _sb_get(f"/team_events?team={team}&year={year}") or [] if isinstance(d, dict)}
    for e in reversed(prior_events(team, event_key)):
        d = by_event.get(e["key"]) or _sb_get(f"/team_event/{team}/{e['key']}")
        x = _sb_fuel(d.get("epa")) if isinstance(d, dict) else None
        if x is not None:
            return SeasonRate(x, _quals_eff(team, event_key, half_life), [e["key"]], "statbotics")
    return None


# ----------------------------------------------------------------------------- blend with the event's matches
def prior_oprs(event_key: str, before: str | None = None, season_weight: float = None, half_life: float = None,
               norm: bool = None, rates: dict | None = None, source: str = None) -> dict[int, float]:
    """{team: fuel per match} for the event's teams: the event's fuel OPR from the quals played before `before`
    (None: none played yet), ridge-shrunk toward each team's season rate. The season rates are put on the
    event's scale by one least-squares factor over the alliance totals played so far (1 before the first);
    a team with no season rate is shrunk toward the event mean with opr.fuel_opr's ridge.
    `rates` ({team: SeasonRate | None}) skips the season_rate lookups."""
    c = SEASON_WEIGHT if season_weight is None else season_weight
    if rates is None:
        rates = {t: season_rate(t, event_key, half_life, norm, source) for t in event_teams(event_key)}
    ms = []
    if before is not None:
        cut = oprmod.match_order(tba.match(before))
        ms = [m for m in tba.event_matches(event_key)
              if m.comp_level == "qm" and m.hub("red") and m.hub("blue") and oprmod.match_order(m) < cut]
    teams = sorted(set(rates) | {t for m in ms for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    have = [t for t in teams if rates.get(t)]
    s = np.array([rates[t].rate if rates.get(t) else np.nan for t in teams])
    rows, y = [], []
    for m in ms:
        for al in ("red", "blue"):
            r = np.zeros(len(teams))
            r[[idx[t] for t in m.teams(al)]] = 1
            rows.append(r)
            y.append(m.hub(al).get("totalCount", 0))
    if not rows:  # event start: the season rates alone (a team without one gets the mean)
        fill = float(np.nanmean(s)) if have else 0.0
        return {t: float(s[idx[t]]) if np.isfinite(s[idx[t]]) else fill for t in teams}
    A, y = np.array(rows), np.array(y, float)
    # scale: season units -> this event's alliance totals (teams without a rate count at the mean rate)
    s_fill = np.where(np.isfinite(s), s, np.nanmean(s) if have else y.mean() / 3)
    pred = A @ s_fill
    k = float(pred @ y / (pred @ pred)) if pred @ pred > 0 else 1.0
    mu = np.where(np.isfinite(s), k * s_fill, y.mean() / 3)
    lam = np.array([c * rates[t].n_eff if rates.get(t) else NO_PRIOR_RIDGE for t in teams])
    lam = np.maximum(lam, 1e-6)
    x = np.linalg.solve(A.T @ A + np.diag(lam), A.T @ (y - A @ mu)) + mu
    return {t: float(max(0.0, x[idx[t]])) for t in teams}


def season_prior(team: int, event_key: str, before_match: str | None = None) -> float | None:
    """A team's fuel scoring rate going into `before_match` at event_key (None for the event start): its season
    rate from earlier events, blended with this event's quals played before that match. None if neither has
    anything on the team."""
    return prior_oprs(event_key, before_match).get(team)
