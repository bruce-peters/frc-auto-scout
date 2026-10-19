"""Lag study, step 2: is the hub-entry -> counter delay constant within a match, or does it stretch in volleys?

For each counter burst (jumps <= 1.5 s apart, >= MIN_BALLS balls) the cumulative counter is compared with the
cumulative hub-top activity (main view + own end view) in a window around it. Quantile lags: when the activity
has reached q of its burst total vs when the counter has; q = 0.1 (onset), 0.5, 0.9 (tail). Also a local
cross-correlation peak per burst. Reported: within-match spread vs between-match spread, and the lag vs the
burst's ball rate (queueing in the hub would make dense bursts lag more at the tail).

  .venv/Scripts/python data/exp_hubentry/tools/lag_bursts.py -> data/exp_hubentry/lag_bursts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lag_study import BIN, OUT, DST, counter_series, feat_series, tl_window  # noqa: E402

MIN_BALLS = 6
GAP = 1.5
PRE, POST = 3.5, 0.0  # activity window: [first jump - PRE - lag0, last jump - lag0 + 1.0]


def bursts(counts):
    out, cur, prev = [], [], 0
    for t, c in counts:
        if c > prev:
            if cur and t - cur[-1][0] > GAP:
                out.append(cur)
                cur = []
            cur.append((t, c - prev))
        prev = max(prev, c)
    if cur:
        out.append(cur)
    return out


def qtime(t, cum, q):
    """first time the cumulative series reaches q of its total (linear interpolation)."""
    tot = cum[-1]
    if tot <= 0:
        return None
    return float(np.interp(q * tot, cum, t))


def main():
    rows = []
    for d in sorted(OUT.iterdir()):
        rp, hp = d / "result.json", d / "hub_feats.json"
        if not rp.exists() or not hp.exists():
            continue
        r = json.loads(rp.read_text())
        tl = r["timeline"]
        lay = tl.get("layout") or "?"
        t0, t1 = tl_window(tl)
        edges = np.arange(t0, t1 + BIN, BIN)
        tc = edges[:-1] + BIN / 2
        hf = {tuple(k.split("|")): np.array(v, dtype=np.float64).reshape(-1, 3)
              for k, v in json.loads(hp.read_text()).items()}
        for al in ("red", "blue"):
            counts = tl["counts"].get(al) or []
            y = counter_series(counts, edges)
            if y.sum() < 20:
                continue
            near = f"{al}_end"
            act = np.zeros(len(tc))
            for v in ("main", near):
                s = feat_series(hf.get((v, al, "top")), edges)
                if s is not None and s.sum() > 0:
                    act += s / max(s.mean(), 1e-9)
            if act.sum() <= 0:
                continue
            for b in bursts(counts):
                n = sum(x[1] for x in b)
                if n < MIN_BALLS:
                    continue
                ta, tb = b[0][0], b[-1][0]
                # activity window: entries ~1.7 s before counting; generous margins
                wa, wb = ta - 1.7 - PRE, tb - 1.7 + 1.2
                m = (tc >= wa) & (tc <= wb)
                if m.sum() < 5:
                    continue
                ca = np.cumsum(act[m])
                # background: subtract the median activity level so idle jitter does not smear quantiles
                bg = np.median(act[(tc >= wa - 6) & (tc < wa)]) if ((tc >= wa - 6) & (tc < wa)).any() else 0.0
                ca = np.cumsum(np.clip(act[m] - bg, 0, None))
                if ca[-1] <= 0:
                    continue
                mc = (tc >= ta - 0.6) & (tc <= tb + 0.3)
                cc = np.cumsum(y[mc])
                row = {"key": d.name, "al": al, "layout": lay, "n": n, "dur": tb - ta,
                       "rate": n / max(tb - ta + 1 / 3, 1 / 3), "t": ta}
                for q in (0.1, 0.5, 0.9):
                    qa, qc = qtime(tc[m], ca, q), qtime(tc[mc], cc, q)
                    row[f"lag{int(q * 100)}"] = None if qa is None or qc is None else qc - qa
                rows.append(row)
        print(d.name, len(rows), flush=True)

    R = rows
    out = {"rows": R}
    for lay in ("chezy", "first"):
        rr = [x for x in R if x["layout"] == lay and x["lag50"] is not None]
        lag = np.array([x["lag50"] for x in rr])
        keys = sorted({(x["key"], x["al"]) for x in rr})
        per = {k: np.array([x["lag50"] for x in rr if (x["key"], x["al"]) == k]) for k in keys}
        means = np.array([v.mean() for v in per.values() if len(v) >= 3])
        within = np.concatenate([v - v.mean() for v in per.values() if len(v) >= 3])
        rate = np.array([x["rate"] for x in rr])
        l10 = np.array([x["lag10"] for x in rr])
        l90 = np.array([x["lag90"] for x in rr])
        big = np.array([x["n"] for x in rr]) >= 15
        s = {"bursts": len(rr), "median_lag50": float(np.median(lag)), "iqr": np.percentile(lag, [25, 75]).tolist(),
             "between_alliance_sd": float(means.std()), "within_alliance_sd": float(within.std()),
             "within_alliance_mad": float(np.median(np.abs(within)) * 1.4826),
             "lag10_median": float(np.median(l10)), "lag90_median": float(np.median(l90)),
             "lag90_minus_lag10_big_bursts": float(np.median((l90 - l10)[big])) if big.any() else None,
             "lag90_minus_lag10_small_bursts": float(np.median((l90 - l10)[~big])),
             "corr_lag50_rate": float(np.corrcoef(lag, rate)[0, 1])}
        # lag vs rate bins
        for lo, hi in ((0, 4), (4, 8), (8, 12), (12, 99)):
            mm = (rate >= lo) & (rate < hi)
            if mm.sum() >= 5:
                s[f"rate_{lo}_{hi}"] = {"n": int(mm.sum()), "lag10": float(np.median(l10[mm])),
                                        "lag50": float(np.median(lag[mm])), "lag90": float(np.median(l90[mm]))}
        out[lay] = s
        print(lay, json.dumps(s, indent=1))
    (DST / "lag_bursts.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
