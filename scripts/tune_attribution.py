"""Re-run only the attribution step from saved results and score it against fuel-OPR shares.

Everything attribution needs is in result.json (overlay counter, shots with team/conf/toward,
team field paths). Parameters are tuned on some events and checked on the others, so the
reported improvement is not just fitting OPR noise.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import attribution, config, field, opr, tba  # noqa: E402

TRAIN_EVENTS = {"2026casnv", "2026joh"}


def load():
    data = []
    oprs = {}
    for f in sorted(config.OUTPUT.glob("*/result.json")):
        r = json.loads(f.read_text())
        info = tba.match(r["match"])
        if info.event_key not in oprs:
            oprs[info.event_key] = opr.fuel_opr(info.event_key)
        data.append((r, info, oprs[info.event_key]))
    return data


def run(r, info, params):
    tl = r["timeline"]
    a0, ts = tl["auto_start"], tl.get("teleop_start")

    def to_mt(t):
        return (20 + t - ts) if ts and t >= ts else t - a0

    def to_vt(mt):
        return (ts + mt - 20) if ts and mt >= 20 else a0 + mt

    out = {}
    for al in ("red", "blue"):
        counts = (tl.get("counts") or {}).get(al)
        if not counts:
            return None
        inc = attribution.increments_from_counts(counts)
        sigs = []
        for team in info.teams(al):
            s = attribution.TeamSignals(team)
            sh = [x for x in r["shots"] if x["team"] == team]
            w = []
            for x in sh:
                c = x["conf"]
                if x.get("toward") is False:
                    c = c / 0.35 * params["away"]  # results were saved with weight 0.35
                w.append(c)
            s.shots = np.array([x["t"] for x in sh])
            s.shot_w = np.array(w)
            pts = r["paths"].get(str(team), [])
            if pts:
                zt = np.array([to_vt(p[0]) for p in pts])
                zv = np.array([1.0 if field.in_alliance_zone(al, p[1], params["margin"]) else 0.0 for p in pts])
                grid = np.arange(zt.min(), zt.max(), 0.5)
                idx = np.clip(np.searchsorted(zt, grid), 0, len(zt) - 1)
                near = np.abs(zt[idx] - grid) < 1.0
                s.zone_t, s.zone_v = grid, np.where(near, zv[idx], np.nan)
            sigs.append(s)
        attribution.PRIOR = params["prior"]
        attribution.LAG_LO, attribution.LAG_PEAK, attribution.LAG_HI = params["lag"]
        if params.get("mode") == "nnls":
            split = attribution.split_nnls(inc, sigs, to_mt, ridge=params.get("ridge", 4.0))
        else:
            split = attribution.split_increments(inc, sigs, to_mt)
        for team in info.teams(al):
            out[team] = sum(split.get(team, {}).values())
    return out


def score(data, params, events):
    """(-share correlation with OPR, uniform-baseline error, n) — lower is better."""
    sa, so = [], []
    errs, base = [], []
    for r, info, o in data:
        if info.event_key not in events:
            continue
        res = run(r, info, params)
        if res is None:
            continue
        for al in ("red", "blue"):
            teams = info.teams(al)
            a = np.array([res[t] for t in teams])
            op = np.array([o.get(t, np.nan) for t in teams])
            if a.sum() < 40 or not np.all(np.isfinite(op)) or op.sum() <= 0:
                continue
            errs += list(np.abs(a / a.sum() - op / op.sum()))
            base += list(np.abs(1 / 3 - op / op.sum()))
            sa += list(a / a.sum())
            so += list(op / op.sum())
    if len(sa) < 6:
        return np.nan, np.nan, 0
    return -float(np.corrcoef(sa, so)[0, 1]), float(np.mean(errs)), len(errs)


def main():
    data = load()
    events = {info.event_key for _, info, _ in data}
    test_events = events - TRAIN_EVENTS
    grid = {
        "prior": [0.05, 0.12, 0.3, 1.0],
        "margin": [0.3, 0.6, 1.2],
        "away": [0.0, 0.35, 1.0],
        "lag": [(0.8, 2.8, 5.5), (0.5, 2.0, 4.5), (1.0, 3.5, 7.0)],
    }
    default = {"prior": 0.12, "margin": 0.6, "away": 0.35, "lag": (0.8, 2.8, 5.5)}
    d_tr = score(data, default, TRAIN_EVENTS)
    d_te = score(data, default, test_events)
    print(f"default: train corr {-d_tr[0]:.3f} (n={d_tr[2]})  test corr {-d_te[0]:.3f} (n={d_te[2]})")
    for ridge in (0.5, 2.0, 8.0, 32.0):
        p = dict(default, mode="nnls", ridge=ridge)
        a, b = score(data, p, TRAIN_EVENTS), score(data, p, test_events)
        print(f"nnls ridge={ridge}: train corr {-a[0]:.3f}  test corr {-b[0]:.3f}", flush=True)
    if "--grid" not in sys.argv:
        return
    best = None
    for vals in itertools.product(*grid.values()):
        p = dict(zip(grid, vals))
        e, _, _ = score(data, p, TRAIN_EVENTS)
        if best is None or e < best[0]:
            best = (e, p)
    e_te = score(data, best[1], test_events)
    print(f"best on train: {best[1]}  train corr {-best[0]:.3f}  ->  test corr {-e_te[0]:.3f} (n={e_te[2]})")
    (config.OUTPUT / "attribution_tuning.json").write_text(json.dumps(
        {"default": {"train": d_tr, "test": d_te}, "best": {"params": best[1], "train": best[0], "test": e_te}},
        default=list, indent=1))


if __name__ == "__main__":
    main()
