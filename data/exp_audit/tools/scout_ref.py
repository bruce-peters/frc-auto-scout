"""5026 scouting (Chezy quals) as a per-team reference for a results dir.

Per team-match: scouted shots (mean over scouts), video fuel, credited launches. Within-alliance share
correlation video vs scouting, next to the fuel-OPR and scout-vs-scout references.
Usage: scout_ref.py SRC [SRC2 ...]   (SRC in common.DIRS: v2, v1, cc_v2)
"""
import json
import sys

import numpy as np

from common import AUDIT, DIRS, load, opr, tba
from autoscout import scouting

srcs = sys.argv[1:] or ["cc_v2"]
rows = scouting.load()
bym = scouting.by_match(rows)


def scout_table(src):
    out = []
    for key, teams in sorted(bym.items()):
        if not (DIRS[src] / key / "result.json").exists():
            continue
        r = load(key, src)
        info = tba.match(key)
        op = opr(info.event_key)
        for al in ("red", "blue"):
            tl = info.teams(al)
            sc = [teams.get(t, []) for t in tl]
            if not all(sc):
                continue  # need every robot of the alliance scouted
            n_launch = {t: sum(1 for s in r["shots"] if s["team"] == t and s["toward"]) for t in tl}
            for t, srow in zip(tl, sc):
                out.append({"match": key, "alliance": al, "team": t,
                            "scout": float(np.mean([x.n for x in srow])),
                            "scout_each": [x.n for x in srow],
                            "video": r["teams"][str(t)]["fuel"], "launches": n_launch[t],
                            "opr": op.get(t, np.nan), "tba_total": info.hub(al).get("totalCount")})
    return out


def shares(tab, col):
    s = []
    by = {}
    for x in tab:
        by.setdefault((x["match"], x["alliance"]), []).append(x)
    for _, xs in by.items():  # same order as `tab` (insertion order)
        v = np.array([x[col] for x in xs], float)
        s += (v / v.sum()).tolist() if v.sum() > 0 else [1 / 3] * len(v)
    return np.array(s)


def scout_pair_shares(tab):
    """share of scout A vs scout B where every robot of the alliance has >= 2 scouts."""
    by = {}
    for x in tab:
        by.setdefault((x["match"], x["alliance"]), []).append(x)
    a, b = [], []
    for _, xs in by.items():
        if all(len(x["scout_each"]) >= 2 for x in xs):
            va = np.array([x["scout_each"][0] for x in xs], float)
            vb = np.array([x["scout_each"][1] for x in xs], float)
            if va.sum() and vb.sum():
                a += (va / va.sum()).tolist(); b += (vb / vb.sum()).tolist()
    return np.array(a), np.array(b)


res = {}
for src in srcs:
    tab = scout_table(src)
    sc, vi, la, op = shares(tab, "scout"), shares(tab, "video"), shares(tab, "launches"), shares(tab, "opr")
    pa, pb = scout_pair_shares(tab)
    n_al = len(tab) // 3
    top_v = sum(int(np.argmax(vi[i:i + 3]) == np.argmax(sc[i:i + 3])) for i in range(0, len(sc), 3))
    top_o = sum(int(np.argmax(op[i:i + 3]) == np.argmax(sc[i:i + 3])) for i in range(0, len(sc), 3))
    m = {"team_matches": len(tab), "alliances": n_al,
         "corr_video_scout": float(np.corrcoef(vi, sc)[0, 1]),
         "corr_launches_scout": float(np.corrcoef(la, sc)[0, 1]),
         "corr_opr_scout": float(np.corrcoef(op, sc)[0, 1]),
         "corr_scoutA_scoutB": float(np.corrcoef(pa, pb)[0, 1]) if len(pa) > 3 else None,
         "n_pair": len(pa),
         "top_video": top_v, "top_opr": top_o,
         "mae_share_video": float(np.mean(np.abs(vi - sc))), "mae_share_opr": float(np.mean(np.abs(op - sc))),
         "mae_share_uniform": float(np.mean(np.abs(1 / 3 - sc)))}
    res[src] = {"metrics": m, "rows": tab}
    print(f"[{src}] {len(tab)} team-matches / {n_al} alliances")
    for k, v in m.items():
        print(f"   {k:22s} {v:.3f}" if isinstance(v, float) else f"   {k:22s} {v}")
    print(f"   {'match':14s} {'al':4s} {'team':>5s} {'scout':>6s} {'video':>6s} {'launch':>6s}  sh_sc sh_vi sh_opr")
    for x, a, b, c in zip(tab, sc, vi, op):
        print(f"   {x['match']:14s} {x['alliance']:4s} {x['team']:5d} {x['scout']:6.0f} {x['video']:6.0f} "
              f"{x['launches']:6d}  {a:5.2f} {b:5.2f} {c:5.2f}{'  <<' if abs(a - b) > 0.2 else ''}")
(AUDIT / ("scout_ref_" + "_".join(srcs) + ".json")).write_text(json.dumps(res, indent=1, default=float))
