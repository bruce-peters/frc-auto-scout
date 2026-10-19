"""Lag study, step 3: impulse responses (NNLS deconvolution) from a signal to the counter increments.

y(t) = sum_k h(k) s(t - k): h is the distribution of delays from a launch / a hub-top blob to a counted ball,
with the signal's own autocorrelation removed. Fitted per alliance-match (signal scaled to the alliance's
ball total, so h sums to ~1 when the signal is complete) and pooled per layout, with a second-difference
smoothness penalty. Width of h = how variable the delay is for single balls (plus detection blur).

  .venv/Scripts/python data/exp_hubentry/tools/lag_deconv.py -> data/exp_hubentry/lag_deconv.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lag_study import BIN, OUT, DST, counter_series, feat_series, hist, tl_window  # noqa: E402

KS = np.arange(-20, 61)  # -2 .. +6 s
SMOOTH = 3.0


def design(s, y):
    n = len(y)
    X = np.zeros((n, len(KS)))
    for j, k in enumerate(KS):
        if k >= 0:
            X[k:, j] = s[:n - k]
        else:
            X[:n + k, j] = s[-k:]
    return X


def solve(Xs, ys, lam=SMOOTH):
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
    if h.sum() <= 0:
        return {}
    p = h / h.sum()
    mu = float((p * lag).sum())
    sd = float(np.sqrt((p * (lag - mu) ** 2).sum()))
    cdf = np.cumsum(p)
    q = [float(np.interp(x, cdf, lag)) for x in (0.1, 0.25, 0.5, 0.75, 0.9)]
    return {"gain": float(h.sum()), "peak": float(lag[int(np.argmax(h))]), "mean": mu, "sd": sd, "q10_25_50_75_90": q}


def main():
    per, pooled = [], {}
    for d in sorted(OUT.iterdir()):
        rp, hp = d / "result.json", d / "hub_feats.json"
        if not rp.exists() or not hp.exists():
            continue
        r = json.loads(rp.read_text())
        tl = r["timeline"]
        lay = tl.get("layout") or "?"
        t0, t1 = tl_window(tl)
        edges = np.arange(t0, t1 + BIN, BIN)
        hf = {tuple(k.split("|")): np.array(v, dtype=np.float64).reshape(-1, 3)
              for k, v in json.loads(hp.read_text()).items()}
        for al in ("red", "blue"):
            y = counter_series(tl["counts"].get(al) or [], edges)
            if y.sum() < 20:
                continue
            sh = [s for s in r["shots"] if s["alliance"] == al and s.get("toward")]
            sigs = {"launch": hist([s["t"] for s in sh], edges, [s["conf"] for s in sh])}
            for v, nm in (("main", "top_main"), (f"{al}_end", "top_near")):
                s = feat_series(hf.get((v, al, "top")), edges)
                if s is not None and s.sum() > 0:
                    sigs[nm] = s
            for v, nm in (("main", "exit_main"), (f"{al}_end", "exit_near")):  # balls leaving the hub's exits
                s = feat_series(hf.get((v, al, "exit")), edges)
                if s is not None and s.sum() > 0:
                    sigs[nm] = s
            if "top_main" in sigs and "top_near" in sigs:
                sigs["top_both"] = sigs["top_main"] / sigs["top_main"].mean() + sigs["top_near"] / sigs["top_near"].mean()
            row = {"key": d.name, "al": al, "layout": lay}
            for nm, s in sigs.items():
                if s.sum() <= 0:
                    continue
                s = s * y.sum() / s.sum()
                X = design(s, y)
                h = solve([X], [y])
                row[nm] = stats(h)
                pooled.setdefault((lay, nm), ([], []))
                pooled[(lay, nm)][0].append(X)
                pooled[(lay, nm)][1].append(y)
            per.append(row)
        print(d.name, *[f"{x['al']} launch {x.get('launch', {}).get('peak')} top {x.get('top_both', {}).get('peak')}"
                        for x in per[-2:]], flush=True)
    summ = {}
    for (lay, nm), (Xs, ys) in sorted(pooled.items()):
        h = solve(Xs, ys)
        st = stats(h)
        pk = [x[nm]["peak"] for x in per if x["layout"] == lay and nm in x and x[nm]]
        sds = [x[nm]["sd"] for x in per if x["layout"] == lay and nm in x and x[nm]]
        st.update({"n": len(Xs), "per_alliance_peak_median": float(np.median(pk)),
                   "per_alliance_peak_iqr": np.percentile(pk, [25, 75]).tolist(),
                   "per_alliance_peak_sd": float(np.std(pk)),
                   "per_alliance_sd_median": float(np.median(sds)),
                   "h": [round(float(x), 5) for x in h]})
        summ[f"{lay}|{nm}"] = st
        print(f"{lay:6s} {nm:9s} pooled peak {st['peak']:.2f} mean {st['mean']:.2f} sd {st['sd']:.2f} q {np.round(st['q10_25_50_75_90'], 2)}"
              f" gain {st['gain']:.2f} | per alliance peak median {st['per_alliance_peak_median']:.2f} sd {st['per_alliance_peak_sd']:.2f}"
              f" width(sd) median {st['per_alliance_sd_median']:.2f}")
    (DST / "lag_deconv.json").write_text(json.dumps({"rows": per, "summary": summ, "lags_s": (KS * BIN).tolist()}, indent=1))


if __name__ == "__main__":
    main()
