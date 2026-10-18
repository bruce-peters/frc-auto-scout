"""Shared helpers for the failure-finder audits (read-only on autoscout/, scripts/, output/).

Rebuilds attribution.TeamSignals from a stored result.json (shots + team paths + overlay counts),
so split_increments can be re-run with other parameters without loading the 190 MB scan.pkl.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import attribution as A, config, field as F, pipeline as P, tba  # noqa: E402

AUDIT = ROOT / "data" / "exp_audit"
# 19:40: the main session installed the v2 rescan into output/ (v1 backed up in data/backup_output_v1)
DIRS = {"v2": ROOT / "output", "v1": ROOT / "data" / "backup_output_v1", "cc_v2": ROOT / "data" / "exp_bumpers" / "robot_v2",
        "rescan_v2": ROOT / "data" / "rescan_v2"}


def match_keys(src: str = "v2") -> list[str]:
    return sorted(p.parent.name for p in DIRS[src].glob("*/result.json"))


def load(key: str, src: str = "v2") -> dict:
    """result.json; with env FUEL_FIELD=fuel_video, teams[*]['fuel'] is replaced by the pure-video split
    (since finalize_v4, 'fuel' is the learned credit model's estimate)."""
    r = json.loads((DIRS[src] / key / "result.json").read_text())
    fld = os.environ.get("FUEL_FIELD")
    if fld:
        for t, v in r["teams"].items():
            if isinstance(v, dict) and fld in v:
                v["fuel"] = v[fld]
    return r


class TL:
    """overlay.Timeline stand-in from result['timeline']."""

    def __init__(self, r):
        t = r["timeline"]
        self.auto_start, self.teleop_start = t["auto_start"], t["teleop_start"]
        self.counts = {k: [tuple(x) for x in v] for k, v in t["counts"].items()}

    def match_time(self, vt):
        if self.auto_start is None:
            return None
        if self.teleop_start is not None and vt >= self.teleop_start:
            return config.AUTO_LEN + (vt - self.teleop_start)
        return vt - self.auto_start

    def video_time(self, mt):
        if mt >= config.AUTO_LEN and self.teleop_start is not None:
            return self.teleop_start + (mt - config.AUTO_LEN)
        return self.auto_start + mt

    def t0(self):
        return (self.auto_start or P.FIRST_AUTO_START) - 1.0

    def t1(self):
        ts = self.teleop_start or ((self.auto_start or P.FIRST_AUTO_START) + P.FIRST_TELEOP_GAP)
        return ts + config.TELEOP_LEN + 5.0


def path_video_times(path, tl: TL) -> np.ndarray:
    """result['paths'][team] rows (match t, x, y) -> (video t, x, y), undoing the auto/teleop gap.

    Points in the pause between auto and teleop have match t >= 20 and come before the teleop clock
    restarts at 20 (a backwards jump in match t); without such a jump every point >= 20 is teleop."""
    mts = [p[0] for p in path]
    jump = next((i for i in range(1, len(mts)) if mts[i] < mts[i - 1] - 0.5), None)
    out = []
    for i, (mt, x, y) in enumerate(path):
        if mt >= config.AUTO_LEN and jump is not None and i < jump:
            vt = tl.auto_start + mt  # in the auto->teleop pause
        else:
            vt = tl.video_time(mt)
        out.append((vt, x, y))
    return np.array(out) if out else np.zeros((0, 3))


def team_signals(r: dict, info, shot_filter=None, zone_margin: float | None = None) -> dict[int, A.TeamSignals]:
    """Same construction as pipeline.analyse's zone-occupancy block (drains off)."""
    tl = TL(r)
    out = {}
    for al in ("red", "blue"):
        for team in info.teams(al):
            ts = A.TeamSignals(team)
            sh = [s for s in r["shots"] if s["team"] == team and (shot_filter is None or shot_filter(s))]
            ts.shots = np.array([s["t"] for s in sh])
            ts.shot_w = np.array([s["conf"] for s in sh])
            pts = path_video_times(r["paths"].get(str(team), []), tl)
            if len(pts):
                pts = pts[np.argsort(pts[:, 0], kind="stable")]
                grid = np.arange(tl.t0(), tl.t1(), 0.5)
                zv = np.full(len(grid), np.nan)
                vv = np.zeros(len(grid))
                for gi, g in enumerate(grid):
                    j = np.searchsorted(pts[:, 0], g)
                    near = [pts[jj] for jj in (j - 1, j) if 0 <= jj < len(pts) and abs(pts[jj][0] - g) < 1.0]
                    vv[gi] = float(any(abs(pts[jj][0] - g) <= P.VIS_WINDOW for jj in (j - 1, j) if 0 <= jj < len(pts)))
                    zk = {} if zone_margin is None else {"margin": zone_margin}
                    if near:
                        zv[gi] = 1.0 if F.in_alliance_zone(al, near[0][1], **zk) else 0.0
                    elif P.FILL_GAPS:
                        p = P._track_position(pts, g)
                        if p is not None:
                            zv[gi] = 1.0 if F.in_alliance_zone(al, p[0], **zk) else 0.0
                ts.zone_t, ts.zone_v = grid, zv
                ts.vis_t, ts.vis_v = grid, vv
            out[team] = ts
    return out


def resplit(r: dict, info, signals=None, log=None, **kw) -> dict[int, float]:
    """Per-team fuel (scaled to TBA like analyse) from the stored overlay increments."""
    tl = TL(r)
    sig = signals or team_signals(r, info, **kw)
    out = {}
    for al in ("red", "blue"):
        if not tl.counts.get(al):
            for t in info.teams(al):
                out[t] = r["teams"][str(t)]["fuel"]
            continue
        inc = A.increments_from_counts(tl.counts[al])
        split = A.split_increments(inc, [sig[t] for t in info.teams(al)], tl.match_time, log=log)
        tot = sum(n for _, n in inc)
        tba_total = info.hub(al).get("totalCount")
        scale = tba_total / tot if tba_total and tot else 1.0
        for t in info.teams(al):
            out[t] = sum(split.get(t, {}).values()) * scale
    return out


_OPR = {}


def opr(event: str) -> dict:
    from autoscout import opr as O

    if event not in _OPR:
        _OPR[event] = O.fuel_opr(event)
    return _OPR[event]


def share_metrics(per_match: dict[str, dict[int, float]]) -> dict:
    """Within-alliance share corr vs fuel OPR, top-scorer hits, MAE (scripts/evaluate.py definitions)."""
    sa, so, ae = [], [], []
    top = n_al = 0
    for key, fuel in per_match.items():
        info = tba.match(key)
        op = opr(info.event_key)
        for al in ("red", "blue"):
            teams = info.teams(al)
            fa = np.array([fuel[t] for t in teams])
            o = np.array([op.get(t, np.nan) for t in teams])
            if fa.sum() < 40 or not np.all(np.isfinite(o)) or o.sum() <= 0:
                continue
            a_, o_ = fa / fa.sum(), o / o.sum()
            sa += a_.tolist(); so += o_.tolist(); ae += (np.abs(a_ - o_) * fa.sum()).tolist()
            top += int(np.argmax(a_) == np.argmax(o_)); n_al += 1
    return {"corr": float(np.corrcoef(sa, so)[0, 1]), "top": top, "alliances": n_al, "mae": float(np.mean(ae))}


def free_gb() -> float:
    import subprocess

    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024 / 1024


_SC = None


def scout_counts() -> dict:
    """{match: {team: mean scouted shots}} for alliances where all three robots were scouted."""
    global _SC
    if _SC is None:
        from autoscout import scouting

        bym = scouting.by_match(scouting.load())
        _SC = {}
        for key, teams in bym.items():
            try:
                info = tba.match(key)
            except Exception:
                continue
            for al in ("red", "blue"):
                tl = info.teams(al)
                if all(teams.get(t) for t in tl):
                    _SC.setdefault(key, {}).update({t: float(np.mean([x.n for x in teams[t]])) for t in tl})
    return _SC


def scout_metrics(per_match: dict[str, dict[int, float]]) -> dict | None:
    """Within-alliance share corr / MAE vs 5026 scouting (Chezy quals) for the matches present."""
    sc = scout_counts()
    a, b = [], []
    for key, fuel in per_match.items():
        if key not in sc:
            continue
        info = tba.match(key)
        for al in ("red", "blue"):
            tl = info.teams(al)
            if not all(t in sc[key] for t in tl):
                continue
            v = np.array([fuel[t] for t in tl]); s = np.array([sc[key][t] for t in tl])
            if v.sum() <= 0 or s.sum() <= 0:
                continue
            a += (v / v.sum()).tolist(); b += (s / s.sum()).tolist()
    if len(a) < 6:
        return None
    a, b = np.array(a), np.array(b)
    return {"scout_corr": float(np.corrcoef(a, b)[0, 1]), "scout_mae_share": float(np.mean(np.abs(a - b))), "n": len(a)}


def time_valid(row) -> bool:
    """Scout row whose timestamps are usable (not a post-hoc entry with every tap in auto or on one
    timestamp): <= 50 shots before 20 s, no timestamp repeated > 40x, >= 3 distinct timestamps."""
    s = row.shots
    if len(s) == 0:
        return True
    _, c = np.unique(s, return_counts=True)
    return bool((s < 20).sum() <= 50 and c.max() <= 40 and len(c) >= 3)


def scout_fuel_mae(per_match: dict[str, dict[int, float]]) -> dict | None:
    """The main session's headline metric: |fuel - TBA alliance total x scouted shot share| per team-match
    (scouted cc quals, every robot scouted)."""
    sc = scout_counts()
    e = []
    for key, fuel in per_match.items():
        if key not in sc:
            continue
        info = tba.match(key)
        for al in ("red", "blue"):
            tl = info.teams(al)
            tot = info.hub(al).get("totalCount")
            if not tot or not all(t in sc[key] for t in tl):
                continue
            s = np.array([sc[key][t] for t in tl])
            if s.sum() <= 0:
                continue
            e += [abs(fuel[t] - tot * s[i] / s.sum()) for i, t in enumerate(tl)]
    return {"scout_fuel_mae": float(np.mean(e)), "n": len(e)} if e else None
