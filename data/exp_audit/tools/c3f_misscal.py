"""Cycle 3f: impact of a missing end-camera calibration, from result.json only.

Per match missing a view: launches seen in that view (team None = could not be linked, because its
boxes have no field position and so no track/team), credited launches per alliance, the alliance's
credit sources, and its share error vs fuel OPR next to all other alliances.
Usage: c3f_misscal.py [v1|v2]
"""
import sys

import numpy as np

from common import A, AUDIT, TL, load, match_keys, opr, tba, team_signals

src = sys.argv[1] if len(sys.argv) > 1 else "v1"
OWN = {"blue_end": "blue", "red_end": "red"}
rows = []
for k in match_keys(src):
    r = load(k, src); info = tba.match(k)
    miss = [v for v in ("blue_end", "red_end") if v not in r["calibration"]]
    op = opr(info.event_key)
    for al in ("red", "blue"):
        teams = info.teams(al)
        f = np.array([r["teams"][str(t)]["fuel"] for t in teams]); o = np.array([op.get(t, np.nan) for t in teams])
        err = float(np.abs(f / f.sum() - o / o.sum()).sum() / 2) if f.sum() > 0 and np.all(np.isfinite(o)) else np.nan
        own_end = f"{al}_end"
        sh = [s for s in r["shots"] if s["alliance"] == al and s["toward"]]
        rows.append({"match": k, "al": al, "missing_own": own_end in miss, "missing_any": bool(miss),
                     "credited": sum(1 for s in sh if s["team"] is not None),
                     "unlinked_own_end": sum(1 for s in sh if s["team"] is None and s["view"] == own_end),
                     "credited_own_end": sum(1 for s in sh if s["team"] is not None and s["view"] == own_end),
                     "share_err": err, "tba": info.hub(al).get("totalCount") or 0})
aff = [x for x in rows if x["missing_own"]]
ok = [x for x in rows if not x["missing_any"]]
print(f"{len(rows) // 2} matches ({src}); alliances whose own end camera is uncalibrated: {len(aff)}")
for x in aff:
    print(f"   {x['match']:18s} {x['al']:4s} credited launches {x['credited']:4d} (own-end view {x['credited_own_end']}), "
          f"own-end launches left unlinked {x['unlinked_own_end']:4d}, share err vs OPR {x['share_err']:.3f}, TBA fuel {x['tba']}")
print(f"   others: credited launches median {np.median([x['credited'] for x in ok]):.0f}, from own-end view median "
      f"{np.median([x['credited_own_end'] for x in ok]):.0f} "
      f"({100 * np.sum([x['credited_own_end'] for x in ok]) / np.sum([x['credited'] for x in ok]):.0f} % of credited launches), "
      f"unlinked own-end median {np.median([x['unlinked_own_end'] for x in ok]):.0f}")
e_a = np.nanmean([x["share_err"] for x in aff]); e_o = np.nanmean([x["share_err"] for x in ok])
print(f"   mean share error vs OPR (total variation): affected {e_a:.3f} vs others {e_o:.3f}")
print(f"   balls in affected alliances: {sum(x['tba'] for x in aff)} of {sum(x['tba'] for x in rows)} "
      f"({100 * sum(x['tba'] for x in aff) / sum(x['tba'] for x in rows):.1f} %)")
