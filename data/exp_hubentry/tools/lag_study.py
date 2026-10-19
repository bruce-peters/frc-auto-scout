"""Lag study, step 1: how far does the scoreboard counter trail hub activity and launches?

Per match and alliance (output/<match>/result.json + hub_feats.json only, no scan.pkl):
  y(t)   counter increments (overlay, 3 Hz), 0.1 s bins, each jump placed mid-way into its 0.33 s read gap
  L(t)   detected launches of the alliance heading for its hub (result shots, toward=True), conf-weighted
  E_v(t) moving-FUEL blobs in the airspace over the alliance's hub opening ("top"), per camera view
  X_v(t) moving-FUEL blobs at the hub's exits ("exit"), per view
Normalised cross-correlation (0.5 s box smoothing) gives a peak lag per pair; peaks are reported per match,
per layout (Chezy vs FIRST) and pooled.

  .venv/Scripts/python data/exp_hubentry/tools/lag_study.py   -> data/exp_hubentry/lag_study.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config  # noqa: E402

OUT = ROOT / "output"
DST = ROOT / "data" / "exp_hubentry"
BIN = 0.1
LAGS = np.arange(-20, 81)  # bins: -2 .. +8 s (positive: counter after the signal)
READ_GAP = 1 / 3.0


def tl_window(tl):
    a0 = tl.get("auto_start") or 6.13
    ts = tl.get("teleop_start") or (a0 + 25.1)
    return a0 - 1.0, ts + config.TELEOP_LEN + 6.0


def counter_series(counts, edges):
    y = np.zeros(len(edges) - 1)
    prev = 0
    for t, c in counts:
        if c > prev:
            i = np.searchsorted(edges, t - READ_GAP / 2) - 1
            if 0 <= i < len(y):
                y[i] += c - prev
        prev = max(prev, c)
    return y


def hist(times, edges, w=None):
    h, _ = np.histogram(times, edges, weights=w)
    return h.astype(float)


def feat_series(arr, edges):
    """mean blobs per frame per bin (frames per bin vary with sub-sampling)."""
    if arr is None or not len(arr):
        return None
    n = hist(arr[:, 0], edges, arr[:, 1])
    c = np.maximum(hist(arr[:, 0], edges), 1)
    return n / c


def smooth(x, w=5):
    return np.convolve(x, np.ones(w) / w, mode="same")


def xcorr(sig, y):
    """Pearson correlation of y(t + lag) with sig(t) for every lag in LAGS."""
    s, yy = smooth(sig), smooth(y)
    out = np.full(len(LAGS), np.nan)
    n = len(s)
    for j, k in enumerate(LAGS):
        if k >= 0:
            a, b = s[:n - k], yy[k:]
        else:
            a, b = s[-k:], yy[:n + k]
        if a.std() > 0 and b.std() > 0:
            out[j] = np.corrcoef(a, b)[0, 1]
    return out


def peak(c):
    if np.all(np.isnan(c)):
        return None, None
    j = int(np.nanargmax(c))
    # parabolic refinement
    if 0 < j < len(c) - 1 and np.isfinite(c[j - 1]) and np.isfinite(c[j + 1]):
        d = c[j - 1] - 2 * c[j] + c[j + 1]
        off = 0.5 * (c[j - 1] - c[j + 1]) / d if d < 0 else 0.0
    else:
        off = 0.0
    return float((LAGS[j] + off) * BIN), float(c[j])


def main():
    rows = []
    pooled = {}
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
            counts = tl["counts"].get(al) or []
            y = counter_series(counts, edges)
            if y.sum() < 20:
                continue
            sh = [s for s in r["shots"] if s["alliance"] == al and s.get("toward")]
            L = hist([s["t"] for s in sh], edges, [s["conf"] for s in sh])
            row = {"key": d.name, "al": al, "layout": lay, "balls": float(y.sum()), "launches": len(sh)}
            sigs = {"launch": L}
            for v in ("main", "blue_end", "red_end"):
                vk = "main" if v == "main" else ("near" if v.startswith(al) else "far")
                for kind in ("top", "exit"):
                    s = feat_series(hf.get((v, al, kind)), edges)
                    if s is not None and s.sum() > 0:
                        sigs[f"{kind}_{vk}"] = s
            for name, s in sigs.items():
                c = xcorr(s, y)
                lag, val = peak(c)
                row[name] = {"lag": lag, "r": val}
                pooled.setdefault((lay, name), []).append(c)
            # launch -> hub top (flight time): correlate top activity with launches
            for name in [n for n in sigs if n.startswith("top_")]:
                c = xcorr(sigs["launch"], sigs[name])
                lag, val = peak(c)
                row[f"flight_{name}"] = {"lag": lag, "r": val}
                pooled.setdefault((lay, f"flight_{name}"), []).append(c)
            rows.append(row)
        print(d.name, lay, *[f"{x['al']}:{x.get('top_near', {}).get('lag')}" for x in rows[-2:]], flush=True)

    summ = {}
    for (lay, name), cs in sorted(pooled.items()):
        C = np.nanmean(np.array(cs), 0)
        lag, val = peak(C)
        per = [x[name]["lag"] for x in rows if x["layout"] == lay and name in x and x[name]["r"] is not None
               and x[name]["r"] > 0.15]
        summ[f"{lay}|{name}"] = {"pooled_lag": lag, "pooled_r": val, "n": len(cs), "n_ok": len(per),
                                 "median": float(np.median(per)) if per else None,
                                 "iqr": [float(np.percentile(per, 25)), float(np.percentile(per, 75))] if per else None,
                                 "sd": float(np.std(per)) if per else None,
                                 "curve": [round(float(x), 4) for x in C]}
    for k, v in summ.items():
        print(f"{k:28s} pooled {v['pooled_lag']} s (r {v['pooled_r']:.3f}) | per-alliance median {v['median']} "
              f"IQR {v['iqr']} sd {v['sd'] and round(v['sd'], 2)} n {v['n_ok']}/{v['n']}")
    (DST / "lag_study.json").write_text(json.dumps({"rows": rows, "summary": summ, "lags_s": (LAGS * BIN).tolist()},
                                                   indent=1))


if __name__ == "__main__":
    main()
