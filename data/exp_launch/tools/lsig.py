"""Per-robot launch-signal candidates for the scouted Chezy quals -> data/exp_launch/sig/<key>.pkl.

Inputs (no output/ reads): data/exp_attrmodel/v5_snapshot/<key>/{result.json,dets.json} (DINO identities),
data/exp_audit/vol_cache_dino/<key>.npz (vol_signal rows on the scans with the v5 labels: t0, be, speed),
5026 scouting. Every signal is a list of point events (video t, weight) per team; the evaluator bins them.

  cred      credited launches (result.json shots with this team)
  rise      rising box-linked flight tracks, speed >= 2 m/s (count)          [vol_signal rows]
  be        their ball-equivalents (mean blob area / one-ball area)          [vol_signal rows]
  drain     hopper drain: per view, the FUEL-yellow fraction inside the robot's box (dets.json fill), median-of-3,
            normalised by that view's p95 fill for the robot, on a 0.25 s grid; drain = positive decrease per
            step; views combined by the max per step
  drainraw  the same without the per-view normalisation
  hub       scoreboard increments n at t_jump, moved back by 2.7 s (median launch->count lag), given to every
            alliance robot in its zone at that time (n / #in-zone robots)
  hubnear   the same, weighted by exp(-dist/3 m) to the own hub instead of the zone indicator
  still     0.5 s grid steps where the robot is in its zone and moved < 0.25 m since the last step

Truth (scout): per team, launch video times of 5026's taps, both mappings kept:
  cont  auto_start + tap - 0.8     (what build_ds.py uses)
  tl    TL.video_time(tap) - 0.8   (skips the auto->teleop pause)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT / "data" / "exp_audit" / "tools"))
from common import TL, path_video_times  # noqa: E402
from autoscout import attribution as A, field as F, pipeline as P, scouting, tba  # noqa: E402

SNAP = ROOT / "data" / "exp_attrmodel" / "v5_snapshot"
VOLC = ROOT / "data" / "exp_audit" / "vol_cache_dino"
OUT = ROOT / "data" / "exp_launch" / "sig"
OUT.mkdir(parents=True, exist_ok=True)
TAP, LAG_HUB, MIN_SPEED = 0.8, 2.7, 2.0


def drain_events(rows_tv, norm: bool):
    """rows_tv: {view: [(t, fill)]} for one team -> [(t, w)]."""
    per_view = []
    for vk, arr in rows_tv.items():
        a = np.array(sorted(arr))
        if len(a) < 6:
            continue
        f = np.array([np.median(a[max(0, i - 1):i + 2, 1]) for i in range(len(a))])
        if norm:
            f = f / max(np.percentile(f, 95), 0.02)
        g = np.arange(np.floor(a[0, 0] * 4) / 4, a[-1, 0], 0.25)
        fi = np.interp(g, a[:, 0], f)
        # only interpolate across short gaps
        j = np.searchsorted(a[:, 0], g)
        gap = a[np.clip(j, 0, len(a) - 1), 0] - a[np.clip(j - 1, 0, len(a) - 1), 0]
        ok = gap <= 0.6
        d = np.maximum(0.0, fi[:-1] - fi[1:]) * (ok[:-1] & ok[1:])
        per_view.append(dict(zip(np.round(g[1:], 2), d)))
    if not per_view:
        return []
    keys = sorted(set().union(*per_view))
    return [(k, max(pv.get(k, 0.0) for pv in per_view)) for k in keys if max(pv.get(k, 0.0) for pv in per_view) > 0]


def main():
    sc_all = scouting.by_match(scouting.load())
    other = "--other" in sys.argv  # unscouted matches (other events) -> sig_other/, no truth
    keys = sorted(p.parent.name for p in SNAP.glob("*/result.json") if p.parent.name not in sc_all) if other else         sorted(k for k in sc_all if (SNAP / k / "result.json").exists())
    out_dir = OUT.parent / "sig_other" if other else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for k in keys:
        info = tba.match(k)
        r = json.loads((SNAP / k / "result.json").read_text())
        rows = json.loads((SNAP / k / "dets.json").read_text())
        tl = TL(r)
        if tl.auto_start is None:
            continue
        sc = sc_all.get(k, {})
        z = np.load(VOLC / f"{k}.npz") if (VOLC / f"{k}.npz").exists() else None
        out = {"key": k, "auto_start": tl.auto_start, "teleop_start": tl.teleop_start, "t0": tl.t0(), "t1": tl.t1(),
               "alliances": {}, "sig": {}, "truth": {}, "scout_n": {}, "timed": {}}
        for al in ("red", "blue"):
            teams = info.teams(al)
            hub = info.hub(al) or {}
            out["alliances"][al] = {"teams": teams, "tba_total": hub.get("totalCount")}
            inc = A.increments_from_counts(P._match_counts(tl, al, hub)) if tl.counts.get(al) else []
            out["alliances"][al]["inc"] = [(float(a), float(b)) for a, b in inc]
            hubc = F.hub_center(al)
            grid = np.arange(tl.t0(), tl.t1(), 0.5)
            zone, dist, pos = {}, {}, {}
            for t in teams:
                pts = path_video_times(r["paths"].get(str(t), []), tl)
                zv, dv, pv = np.full(len(grid), np.nan), np.full(len(grid), np.nan), np.full((len(grid), 2), np.nan)
                if len(pts):
                    pts = pts[np.argsort(pts[:, 0], kind="stable")]
                    for gi, g in enumerate(grid):
                        p = P._track_position(pts, g)
                        if p is not None:
                            pv[gi] = p[:2]
                            zv[gi] = 1.0 if F.in_alliance_zone(al, p[0]) else 0.0
                            dv[gi] = float(np.hypot(p[0] - hubc[0], p[1] - hubc[1]))
                zone[t], dist[t], pos[t] = zv, dv, pv
            for t in teams:
                s = {}
                s["cred"] = [(x["t"], 1.0) for x in r["shots"] if x["team"] == t]
                if z is not None and str(t) in z.files:
                    a = z[str(t)]
                    a = a[a[:, 2] >= MIN_SPEED]
                    s["rise"] = [(float(x[0]), 1.0) for x in a]
                    s["be"] = [(float(x[0]), float(x[1])) for x in a]
                else:
                    s["rise"], s["be"] = [], []
                tv = {}
                for d in rows:
                    if d[7] == t and d[9] is not None:
                        tv.setdefault(d[1], []).append((d[0], d[9]))
                s["drain"] = drain_events(tv, True)
                s["drainraw"] = drain_events(tv, False)
                # hub-side: increments back-shifted, split over in-zone / near robots
                hz, hn = [], []
                for tj, n in inc:
                    tl_ = tj - LAG_HUB
                    gi = int(np.clip(np.searchsorted(grid, tl_), 0, len(grid) - 1))
                    zs = {u: zone[u][gi] for u in teams}
                    nz = sum(1 for u in teams if zs[u] == 1.0)
                    if zs[t] == 1.0 and nz:
                        hz.append((tl_, n / nz))
                    ws = {u: (np.exp(-dist[u][gi] / 3.0) if np.isfinite(dist[u][gi]) else 0.0) for u in teams}
                    if ws[t] > 0:
                        hn.append((tl_, n * ws[t] / sum(ws.values())))
                s["hub"], s["hubnear"] = hz, hn
                p = pos[t]
                step = np.r_[np.nan, np.hypot(*(p[1:] - p[:-1]).T)]
                s["still"] = [(float(g), 1.0) for g, zz, st in zip(grid, zone[t], step) if zz == 1.0 and st < 0.25]
                out["sig"][t] = s
                rws = sc.get(t, [])
                out["scout_n"][t] = float(np.mean([x.n for x in rws])) if rws else None
                timed = [x for x in rws if x.time_valid]
                out["timed"][t] = len(timed)
                if not timed:
                    out["truth"][t] = {"cont": [], "tl": []}
                    continue
                out["truth"][t] = {
                    "cont": [(float(tl.auto_start + v - TAP), 1.0 / len(timed)) for x in timed for v in x.shots],
                    "tl": [(float(tl.video_time(float(v)) - TAP), 1.0 / len(timed)) for x in timed for v in x.shots]}
        with open(out_dir / f"{k}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        n_ok += 1
        print(k, {t: {kk: len(v) for kk, v in s.items()} for t, s in list(out["sig"].items())[:1]}, flush=True)
    print(n_ok, "matches")


if __name__ == "__main__":
    main()
