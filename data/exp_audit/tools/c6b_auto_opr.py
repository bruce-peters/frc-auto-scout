"""Cycle 7b: auto vs teleop attribution against period-specific OPR.

auto OPR = least squares (ridge, like autoscout.opr.fuel_opr) on TBA hubScore.autoCount over the event's
quals; teleop OPR on totalCount - autoCount. Within-alliance share corr of video fuel_by_period vs those,
next to the uniform split. Also how auto credit splits between launch / prior / hidden by robot visibility.
Usage: c6b_auto_opr.py [v2]
"""
import sys

import numpy as np

from common import load, match_keys, tba

src = sys.argv[1] if len(sys.argv) > 1 else "v2"


def period_opr(event, fn, ridge=0.5):
    ms = [m for m in tba.event_matches(event) if m.comp_level == "qm" and m.breakdown]
    teams = sorted({t for m in ms for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    rows, y = [], []
    for m in ms:
        for al in ("red", "blue"):
            h = m.hub(al)
            if not h:
                continue
            r = np.zeros(len(teams))
            for t in m.teams(al):
                r[idx[t]] = 1
            rows.append(r); y.append(fn(h))
    A_, y = np.array(rows), np.array(y, float)
    mu = y.mean() / 3
    x = np.linalg.solve(A_.T @ A_ + ridge * np.eye(len(teams)), A_.T @ (y - A_ @ np.full(len(teams), mu))) + mu
    return {t: max(0.0, x[idx[t]]) for t in teams}


if __name__ == "__main__":
    OP = {}
    res = {"auto": ([], [], []), "teleop": ([], [], [])}
    for k in match_keys(src):
        r = load(k, src); info = tba.match(k)
        ev = info.event_key
        if ev not in OP:
            OP[ev] = (period_opr(ev, lambda h: h.get("autoCount", 0)),
                      period_opr(ev, lambda h: h.get("totalCount", 0) - h.get("autoCount", 0)))
        oa, ot = OP[ev]
        for al in ("red", "blue"):
            tm = info.teams(al)
            fa = np.array([r["teams"][str(t)]["fuel_by_period"].get("auto", 0.0) for t in tm])
            ft = np.array([sum(v for p, v in r["teams"][str(t)]["fuel_by_period"].items() if p != "auto") for t in tm])
            for name, f, o, minf in (("auto", fa, np.array([oa.get(t, np.nan) for t in tm]), 15),
                                     ("teleop", ft, np.array([ot.get(t, np.nan) for t in tm]), 40)):
                if f.sum() < minf or not np.all(np.isfinite(o)) or o.sum() <= 0:
                    continue
                res[name][0].extend(f / f.sum()); res[name][1].extend(o / o.sum()); res[name][2].append(f.sum())
    for name, (v, o, tot) in res.items():
        v, o = np.array(v), np.array(o)
        top = sum(int(np.argmax(v[i:i + 3]) == np.argmax(o[i:i + 3])) for i in range(0, len(v), 3))
        print(f"{name:6s}: {len(v) // 3} alliances, share corr vs {name}-OPR {np.corrcoef(v, o)[0, 1]:.3f}, "
              f"MAE share video {np.mean(np.abs(v - o)):.3f} vs uniform {np.mean(np.abs(1 / 3 - o)):.3f}, "
              f"top {top}/{len(v) // 3}, mean alliance {name} fuel {np.mean(tot):.0f}")
