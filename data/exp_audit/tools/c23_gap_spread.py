"""Cycle 23: counter-reading gaps. When the overlay counter is not read for >= G s and then jumps by n balls,
increments_from_counts puts all n at the jump time; the kernel then credits whoever launched just before
the jump. Variant: spread such an increment uniformly over (prev reading, jump] in 0.33 s steps.
Readings > 15 s after teleop end are dropped in all variants (mirrors pipeline._match_counts).
Scored vs scouting (primary) and fuel OPR.
"""
import numpy as np

from common import A, TL, config, load, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals

R, INFO, SIG = {}, {}, {}
for k in match_keys("v2"):
    R[k] = load(k, "v2"); INFO[k] = tba.match(k); SIG[k] = team_signals(R[k], INFO[k])
ORIG_INC = A.increments_from_counts


def make_inc(gap_min, n_min, end_t):
    def inc_fn(counts):
        counts = [(t, c) for t, c in counts if end_t is None or t <= end_t]
        out = []
        prev, prev_t = 0, None
        for t, c in counts:
            if c > prev:
                n = float(c - prev)
                if prev_t is not None and t - prev_t >= gap_min and n >= n_min:
                    ts = np.arange(prev_t + 0.33, t + 1e-6, 0.33)
                    out += [(float(x), n / len(ts)) for x in ts]
                else:
                    out.append((t, n))
            prev = max(prev, c)
            prev_t = t
        return out
    return inc_fn


res = {}
for name, gap_min, n_min in (("lump (end-filtered)", 1e9, 1e9), ("spread gap>=2 n>=6", 2.0, 6), ("spread gap>=2 n>=3", 2.0, 3),
                             ("spread gap>=4 n>=6", 4.0, 6), ("spread gap>=1 n>=6", 1.0, 6)):
    pm = {}
    for k in R:
        tl = TL(R[k])
        end_t = tl.teleop_start + config.TELEOP_LEN + 15.0 if tl.teleop_start else None
        A.increments_from_counts = make_inc(gap_min, n_min, end_t)
        pm[k] = resplit(R[k], INFO[k], signals=SIG[k])
        A.increments_from_counts = ORIG_INC
    res[name] = pm
    print(f"{name:20s}", share_metrics(pm), scout_metrics(pm), flush=True)
