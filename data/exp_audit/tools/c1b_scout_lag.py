"""Cycle 1b: timing check against the Chezy scouting.

Scouted shot times (s from match start) vs (a) the overlay counter increments and (b) the video's
detected launches of the same robot. Two clock hypotheses for the scout timer:
  'cont'  : video t = auto_start + scout t (timer keeps running through the auto->teleop pause)
  'match' : video t = Timeline.video_time(scout t) (timer follows the match clock)
Usage: c1b_scout_lag.py [SRC]   (default cc_v2)
"""
import json
import sys

import numpy as np
from scipy.optimize import nnls

from common import A, AUDIT, DIRS, TL, load, tba
from autoscout import scouting

src = sys.argv[1] if len(sys.argv) > 1 else "cc_v2"
BIN = 0.2
LAGS = np.arange(-25, 51)
bym = scouting.by_match(scouting.load())


def to_video(tl, st, mode):
    st = np.asarray(st, float)
    if mode == "cont":
        return tl.auto_start + st
    return np.array([tl.video_time(x) for x in st])


def deconv(pairs):
    Xs, ys = [], []
    for src_t, y in pairs:
        nb = len(y)
        L = src_t * (y.sum() / max(src_t.sum(), 1e-9))
        X = np.zeros((nb, len(LAGS)))
        for j, l in enumerate(LAGS):
            if l >= 0:
                X[l:, j] = L[:nb - l]
            else:
                X[:nb + l, j] = L[-l:]
        Xs.append(X); ys.append(y)
    X, y = np.vstack(Xs), np.concatenate(ys)
    n = X.shape[1]
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i:i + 3] = (1, -2, 1)
    lam = 0.3 * np.sqrt((X ** 2).sum() / n)
    h, _ = nnls(np.vstack([X, lam * D]), np.concatenate([y, np.zeros(n - 2)]))
    return h


def stats(h):
    lag = LAGS * BIN
    cdf = np.cumsum(h) / h.sum()
    q = lambda p: round(float(lag[np.searchsorted(cdf, p)]), 1)
    return {"mode": round(float(lag[np.argmax(h)]), 1), "q10": q(.1), "q25": q(.25), "median": q(.5), "q75": q(.75),
            "q90": q(.9), "peak_frac_1s": round(float(h[np.abs(lag - lag[np.argmax(h)]) <= 0.5].sum() / h.sum()), 3)}


def hist(times, t0, nb):
    y = np.zeros(nb)
    for t in times:
        i = int((t - t0) / BIN)
        if 0 <= i < nb:
            y[i] += 1
    return y


out = {}
for mode in ("cont", "match"):
    pc, pl = [], []  # (scout hist, counter hist), (scout hist, launch hist) pairs
    for key, teams in sorted(bym.items()):
        if not (DIRS[src] / key / "result.json").exists():
            continue
        r = load(key, src)
        info = tba.match(key)
        tl = TL(r)
        t0, t1 = tl.t0(), tl.t1() + 5
        nb = int((t1 - t0) / BIN) + 1
        for al in ("red", "blue"):
            rows = [x for t in info.teams(al) for x in teams.get(t, [])]
            if not rows or not tl.counts.get(al):
                continue
            inc = A.increments_from_counts(tl.counts[al])
            yc = np.zeros(nb)
            for t, n in inc:
                i = int((t - t0) / BIN)
                if 0 <= i < nb:
                    yc[i] += n
            # alliance scout histogram: each robot's mean over its scouts
            sc = np.zeros(nb)
            for t in info.teams(al):
                rs = teams.get(t, [])
                for x in rs:
                    sc += hist(to_video(tl, x.shots, mode), t0, nb) / len(rs)
                launches = [s["t"] for s in r["shots"] if s["team"] == t and s["toward"]]
                if rs and len(launches) >= 5:
                    for x in rs:
                        if x.n >= 5:
                            pl.append((hist(to_video(tl, x.shots, mode), t0, nb), hist(launches, t0, nb)))
            if sc.sum() >= 10:
                pc.append((sc, yc))
    hc, hl = deconv(pc), deconv(pl)
    out[mode] = {"scout_to_counter": stats(hc), "scout_to_launch": stats(hl), "n_alliances": len(pc),
                 "n_scout_rows": len(pl), "h_counter": hc.round(4).tolist(), "h_launch": hl.round(4).tolist()}
    print(f"[{mode}] scout -> counter ({len(pc)} alliances): {out[mode]['scout_to_counter']}")
    print(f"[{mode}] scout -> detected launch ({len(pl)} scout rows): {out[mode]['scout_to_launch']}")
(AUDIT / f"c1b_scout_lag_{src}.json").write_text(json.dumps(out, indent=1))
