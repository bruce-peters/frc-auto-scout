"""Where are robots (and their launches) in auto vs their zone line? And zone gating off per period."""
import sys
from collections import Counter
import numpy as np
from common import A, F, P, TL, load, match_keys, path_video_times, resplit, share_metrics, tba, team_signals
src = sys.argv[1] if len(sys.argv) > 1 else "v2"
ZL = {"blue": F.ALLIANCE_ZONE_DEPTH, "red": F.L - F.ALLIANCE_ZONE_DEPTH}
dep = {"auto": [], "teleop": []}; zst = {"auto": Counter(), "teleop": Counter()}
R, INFO, SIG = {}, {}, {}
for k in match_keys(src):
    r = load(k, src); info = tba.match(k); tl = TL(r)
    R[k], INFO[k] = r, info; SIG[k] = team_signals(r, info)
    for s in r["shots"]:
        if s["team"] is None or not s["toward"]:
            continue
        mt = tl.match_time(s["t"])
        per = "auto" if mt is not None and mt < 20 else "teleop"
        al = "red" if s["team"] in info.red else "blue"
        ts = SIG[k][s["team"]]
        m = (ts.zone_t >= s["t"] - 0.25) & (ts.zone_t <= s["t"] + 0.25)
        zv = ts.zone_v[m]
        zst[per]["unknown" if not len(zv) or np.isnan(zv).all() else ("in" if np.nanmean(zv) >= 0.5 else "out")] += 1
        pts = path_video_times(r["paths"].get(str(s["team"]), []), tl)
        p = P._track_position(pts, s["t"]) if len(pts) else None
        if p is not None:
            dep[per].append((ZL["blue"] - p[0]) if al == "blue" else (p[0] - ZL["red"]))
for per in ("auto", "teleop"):
    d = np.array(dep[per]); n = sum(zst[per].values())
    print(f"{per:6s} credited launches {n}: zone at launch {dict(zst[per])};  depth into own zone (m) quantiles "
          f"10/25/50/75/90 {np.round(np.quantile(d, [.1, .25, .5, .75, .9]), 2)};  share 0..-1.5 m outside {np.mean((d < 0) & (d > -1.5)):.3f}")
# zone gating off in teleop only / everywhere
zf = A.TeamSignals.zone_frac
def run(off_auto, off_tele):
    out = {}
    for k in R:
        tl = TL(R[k])
        def z(self, a, b, tl=tl):
            mt = tl.match_time(b + A.LAG_LO - 2.0)
            auto = mt is not None and A.period_of(mt) == "auto"
            return 1.0 if ((auto and off_auto) or (not auto and off_tele)) else zf(self, a, b)
        A.TeamSignals.zone_frac = z
        out[k] = resplit(R[k], INFO[k], signals=SIG[k])
    A.TeamSignals.zone_frac = zf
    return share_metrics(out)
for name, a, b in (("gating as is", False, False), ("off in auto", True, False), ("off in teleop", False, True), ("off everywhere", True, True)):
    print(f"   {name:15s}", run(a, b))
