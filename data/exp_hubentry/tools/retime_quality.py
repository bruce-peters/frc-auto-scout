"""Entry re-timing quality (all 131 processed matches, cache + output/ only).

Each counted ball is re-timed to an entry time (hubentry.retime_queue) from evidence in some views, then checked
against evidence it did NOT use:
  holdout  re-time from the main view only (entry tracks + blobs); impulse response from the own-end view's entry
           tracks to the re-timed balls. Good re-timing: peak at ~0 s and narrow. Compared with the raw counter
           (peak ~1.3 s, wide) and a constant shift (counter - 1.6 s).
  flight   impulse response from detected launches (toward the hub) to the re-timed balls (all views used): the
           launch -> entry flight time. Compared with launches -> counter (today's kernel problem).
Spread is summarised by the IR's interquartile range and the share of its mass within +-0.5 s of its peak.

  .venv/Scripts/python data/exp_hubentry/tools/retime_quality.py -> data/exp_hubentry/retime_quality.json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from lag_study import BIN, OUT, DST, counter_series, hist, tl_window  # noqa: E402

from autoscout import hubentry as H  # noqa: E402

KS = np.arange(-20, 51)  # -2 .. +5 s


def design(s, y):
    n = len(y)
    X = np.zeros((n, len(KS)))
    for j, k in enumerate(KS):
        if k >= 0:
            X[k:, j] = s[:n - k]
        else:
            X[:n + k, j] = s[-k:]
    return X


def solve(Xs, ys, lam=3.0):
    X, y = np.vstack(Xs), np.concatenate(ys)
    m = len(KS)
    D = np.zeros((m - 2, m))
    for i in range(m - 2):
        D[i, i:i + 3] = (1, -2, 1)
    sc = np.sqrt(lam * max(1.0, (X ** 2).sum() / m / 50))
    h, _ = nnls(np.vstack([X, sc * D]), np.concatenate([y, np.zeros(m - 2)]))
    return h


def stats(h):
    lag = KS * BIN
    p = h / max(h.sum(), 1e-12)
    j = int(np.argmax(h))
    cdf = np.cumsum(p)
    q = [float(np.interp(x, cdf, lag)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
    near = float(p[np.abs(lag - lag[j]) <= 0.5 + 1e-9].sum())
    return {"peak": float(lag[j]), "q10_25_50_75_90": [round(x, 2) for x in q], "iqr": round(q[3] - q[1], 2),
            "mass_within_0.5s_of_peak": round(near, 3), "gain": round(float(h.sum()), 3)}


def series_from(entries, edges):
    return hist([e.entry_t for e in entries], edges, [e.n for e in entries])


pool: dict = {}
for cp in sorted((DST / "cache").glob("*.pkl")):
    key = cp.stem
    r = json.loads((OUT / key / "result.json").read_text())
    tl = r["timeline"]
    lay = tl.get("layout") or "?"
    t0, t1 = tl_window(tl)
    edges = np.arange(t0, t1 + BIN, BIN)
    c = pickle.loads(cp.read_bytes())
    hf = {tuple(k.split("|")): np.array(v, dtype=np.float64).reshape(-1, 3)
          for k, v in json.loads((OUT / key / "hub_feats.json").read_text()).items()}
    for al in ("red", "blue"):
        inc, prev = [], 0
        for t, n in tl["counts"].get(al) or []:
            if n > prev and t0 <= t <= t1:
                inc.append((t, float(n - prev)))
            prev = max(prev, n)
        if sum(n for _, n in inc) < 20:
            continue
        y_cnt = counter_series(tl["counts"].get(al) or [], edges)
        # holdout view: own end camera entry tracks
        Tn = c["views"].get(f"{al}_end")
        if Tn is not None and len(Tn):
            hold = hist(Tn[H.is_entry(Tn, al), H.C["t1"]], edges)
        else:
            hold = np.zeros(len(edges) - 1)
        main_only = {"main": c["views"].get("main")}
        hf_main = {k: v for k, v in hf.items() if k[0] == "main"}
        rt, rate = H.entry_series(main_only, hf_main, al, t0, t1)
        ent_main = H.retime_queue(inc, rt, rate)
        rt2, rate2 = H.entry_series(c["views"], hf, al, t0, t1)
        ent_all = H.retime_queue(inc, rt2, rate2)
        ent_const = [H.Retimed(t, t - H.READ_GAP / 2 - 1.6, n, 1.0) for t, n in inc]
        sh = [s for s in r["shots"] if s["alliance"] == al and s.get("toward")]
        L = hist([s["t"] for s in sh], edges, [s["conf"] for s in sh])
        cov = np.mean([e.conf > 0 for e in ent_all])
        pool.setdefault((lay, "coverage"), []).append(cov)
        targets = {"counter": y_cnt, "const1.6": series_from(ent_const, edges), "retimed_main": series_from(ent_main, edges),
                   "retimed_all": series_from(ent_all, edges)}
        for tn, y in targets.items():
            for sn, s in (("holdout_end_tracks", hold), ("launches", L)):
                if sn == "holdout_end_tracks" and tn == "retimed_all":
                    continue  # not held out
                if sn == "launches" and tn == "retimed_main":
                    continue
                if s.sum() < 5:
                    continue
                ss = s * y.sum() / s.sum()
                pool.setdefault((lay, sn, tn), ([], []))
                pool[(lay, sn, tn)][0].append(design(ss, y))
                pool[(lay, sn, tn)][1].append(y)
    print(key, flush=True)

summ = {}
for k, v in sorted(pool.items(), key=lambda x: str(x[0])):
    if k[1] == "coverage":
        summ[f"{k[0]}|coverage"] = float(np.mean(v))
        print(k, "share of counted readings re-timed from evidence", round(float(np.mean(v)), 3))
        continue
    st = stats(solve(*v))
    st["n"] = len(v[0])
    summ["|".join(k)] = st
    print(f"{k[0]:6s} {k[1]:20s} -> {k[2]:13s} peak {st['peak']:+.2f} q {st['q10_25_50_75_90']} IQR {st['iqr']:.2f} "
          f"mass+-0.5 {st['mass_within_0.5s_of_peak']:.2f} gain {st['gain']:.2f}")
(DST / "retime_quality.json").write_text(json.dumps(summ, indent=1))
