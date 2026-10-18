"""Auto-only zone-gating variants through split_increments (teleop unchanged):
  as_is        : w = z_window * (s + prior + hidden)
  per_launch   : w = sum_k kernel*conf*z(launch_k) + z_window * (prior + hidden)
  launch_ungated: w = s + z_window * (prior + hidden)
  off          : w = s + prior + hidden
"""
import sys
import numpy as np
from c6b_auto_opr import period_opr
from common import A, TL, load, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals
ORIG = A.split_increments


def zone_at(ts, t):
    m = (ts.zone_t >= t - 0.5) & (ts.zone_t <= t + 0.5)
    v = ts.zone_v[m]; v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else np.nan


def make(mode, tl):
    def split(increments, teams, to_match_time, log=None):
        out = {ts.team: {} for ts in teams}
        zl = {ts.team: np.array([zone_at(ts, t) for t in ts.shots]) if len(ts.shots) else np.zeros(0) for ts in teams}
        for t, n in increments:
            if n <= 0:
                continue
            mt = to_match_time(t - 2.0)
            auto = mt is not None and A.period_of(mt) == "auto"
            w = np.zeros(len(teams))
            for i, ts in enumerate(teams):
                z = ts.zone_frac(t - A.LAG_HI, t - A.LAG_LO)
                kw = (A.kernel(t - ts.shots) * ts.shot_w) if len(ts.shots) else np.zeros(0)
                s = float(kw.sum())
                vis = ts.vis_frac(t - A.LAG_HI, t - A.LAG_LO)
                ph = A.PRIOR * vis + A.HIDDEN * (1.0 - vis)
                if not auto or mode == "as_is":
                    w[i] = z * (s + ph)
                elif mode == "per_launch":
                    w[i] = float((kw * np.nan_to_num(zl[ts.team], nan=z)).sum()) + z * ph
                elif mode == "launch_ungated":
                    w[i] = s + z * ph
                elif mode == "off":
                    w[i] = s + ph
            if w.sum() <= 1e-9:
                w = np.ones(len(teams))
            w = w / w.sum()
            per = A.period_of(mt) if mt is not None else "unknown"
            for i, ts in enumerate(teams):
                out[ts.team][per] = out[ts.team].get(per, 0.0) + n * w[i]
        return out
    return split


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "v2"
    R, INFO, SIG = {}, {}, {}
    for k in match_keys(src):
        R[k] = load(k, src); INFO[k] = tba.match(k); SIG[k] = team_signals(R[k], INFO[k])
    OPA = {}
    for mode in ("as_is", "per_launch", "launch_ungated", "off"):
        pm, va, vo = {}, [], []
        for k in R:
            A.split_increments = make(mode, TL(R[k]))
            pm[k] = resplit(R[k], INFO[k], signals=SIG[k])
            # auto shares vs auto-OPR
            tl = TL(R[k]); info = INFO[k]
            if info.event_key not in OPA:
                OPA[info.event_key] = period_opr(info.event_key, lambda h: h.get("autoCount", 0))
            for al in ("red", "blue"):
                if not tl.counts.get(al):
                    continue
                inc = [(t, n) for t, n in A.increments_from_counts(tl.counts[al])
                       if tl.match_time(t - 2.0) is not None and A.period_of(tl.match_time(t - 2.0)) == "auto"]
                sp = A.split_increments(inc, [SIG[k][t] for t in info.teams(al)], tl.match_time)
                v = np.array([sum(sp.get(t, {}).values()) for t in info.teams(al)])
                o = np.array([OPA[info.event_key].get(t, np.nan) for t in info.teams(al)])
                if v.sum() >= 15 and np.all(np.isfinite(o)) and o.sum() > 0:
                    va += (v / v.sum()).tolist(); vo += (o / o.sum()).tolist()
        A.split_increments = ORIG
        va, vo = np.array(va), np.array(vo)
        print(f"{mode:15s} auto corr {np.corrcoef(va, vo)[0, 1]:.3f} MAE {np.mean(np.abs(va - vo)):.3f} | total", share_metrics(pm), scout_metrics(pm))
