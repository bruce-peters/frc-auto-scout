"""Cycle 19: launches made while the shooter's hub is inactive.

A ball launched more than ~FLIGHT s before its hub turns active lands while it is still inactive and cannot
score, but the kernel lets that launch claim counter increments in the first seconds after activation.
(1) share of credited launch evidence (kernel-weighted, ball-weighted) that comes from launches made in an
    inactive period of the shooter's alliance (excluding the last FLIGHT s before activation)
(2) variant: those launches get weight 0 (split otherwise identical to split_increments)
Scored vs scouting (primary) and fuel OPR.
"""
import numpy as np

from common import A, TL, load, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals
from autoscout.config import AUTO_LEN, TELEOP_PERIODS

R, INFO, SIG, INACT = {}, {}, {}, {}
for k in match_keys("v2"):
    R[k] = load(k, "v2"); INFO[k] = tba.match(k); SIG[k] = team_signals(R[k], INFO[k])
    INACT[k] = tba.inactive_periods(INFO[k]) if INFO[k].breakdown else {"red": set(), "blue": set()}
SPAN = {name: (AUTO_LEN + a, AUTO_LEN + b) for name, a, b in TELEOP_PERIODS}
ORIG = A.split_increments


def dead_launch(mt, inactive, flight):
    """launch at match time mt cannot score: inside an inactive period and > flight s before its end."""
    for p in inactive:
        a, b = SPAN[p]
        if a <= mt < b - flight:
            return True
    return False


def make(k, al_of, flight, scale=0.0):
    tl = TL(R[k])

    def split(increments, teams, to_match_time, log=None):
        out = {ts.team: {} for ts in teams}
        masks = {}
        for ts in teams:
            inact = INACT[k][al_of[ts.team]]
            masks[ts.team] = np.array([scale if (tl.match_time(t) is not None and dead_launch(tl.match_time(t), inact, flight)) else 1.0
                                       for t in ts.shots]) if len(ts.shots) else np.zeros(0)
        for t, n in increments:
            if n <= 0:
                continue
            mt = to_match_time(t - 2.0)
            per = A.period_of(mt) if mt is not None else "unknown"
            gate = A.AUTO_ZONE_GATE or per != "auto"
            w = np.zeros(len(teams))
            for i, ts in enumerate(teams):
                z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO) if gate else 1.0
                s = float((A.kernel(t - ts.shots) * ts.shot_w * masks[ts.team]).sum()) if len(ts.shots) else 0.0
                vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                w[i] = z * (s + A.PRIOR * vis + A.HIDDEN * (1.0 - vis))
            if w.sum() <= 1e-9:
                w = np.ones(len(teams))
            w = w / w.sum()
            for i, ts in enumerate(teams):
                out[ts.team][per] = out[ts.team].get(per, 0.0) + n * w[i]
        return out
    return split


# (1) share of launch evidence from dead launches
dead_e = tot_e = 0.0
dead_n = tot_n = 0
for k in R:
    r, info = R[k], INFO[k]; tl = TL(r)
    for al in ("red", "blue"):
        inact = INACT[k][al]
        for s in r["shots"]:
            if s["team"] is not None and s["toward"] and s["alliance"] == al:
                mt = tl.match_time(s["t"])
                tot_n += 1
                dead_n += int(mt is not None and dead_launch(mt, inact, 1.5))
        if not tl.counts.get(al):
            continue
        for t, n in A.increments_from_counts(tl.counts[al]):
            for team in info.teams(al):
                ts = SIG[k][team]
                if not len(ts.shots):
                    continue
                kw = A.kernel(t - ts.shots) * ts.shot_w
                dm = np.array([tl.match_time(x) is not None and dead_launch(tl.match_time(x), inact, 1.5) for x in ts.shots])
                tot_e += n * kw.sum(); dead_e += n * kw[dm].sum()
print(f"{len(R)} matches: credited launches made while own hub inactive (> 1.5 s before activation): {dead_n} / {tot_n} "
      f"({100 * dead_n / tot_n:.1f} %); share of ball-weighted kernel evidence they carry: {100 * dead_e / max(tot_e, 1e-9):.1f} %")
base = {k: resplit(R[k], INFO[k], signals=SIG[k]) for k in R}
print("base          ", share_metrics(base), scout_metrics(base))
for flight in (1.0, 1.5, 2.5):
    pm = {}
    for k in R:
        info = INFO[k]
        al_of = {t: ("red" if t in info.red else "blue") for t in info.red + info.blue}
        A.split_increments = make(k, al_of, flight)
        pm[k] = resplit(R[k], info, signals=SIG[k])
        A.split_increments = ORIG
    print(f"drop dead (flight {flight}):", share_metrics(pm), scout_metrics(pm), flush=True)
