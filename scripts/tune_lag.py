"""Estimate the launch -> counter-increment delay from processed matches.

Cross-correlates, per alliance, the time series of detected launches (all of the alliance's
robots) with the overlay counter increments. The peak gives the typical delay (flight +
hub sensor + display lag) used by the attribution kernel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config  # noqa: E402

BIN = 0.25


def main():
    lags = np.arange(-2.0, 8.0, BIN)
    acc = np.zeros(len(lags))
    n = 0
    for f in sorted(config.OUTPUT.glob("*/result.json")):
        r = json.loads(f.read_text())
        tl = r["timeline"]
        for al in ("red", "blue"):
            counts = (tl.get("counts") or {}).get(al) or []
            if len(counts) < 5:
                continue
            shots = np.array([s["t"] for s in r["shots"] if s.get("alliance") == al])
            if len(shots) < 10:
                continue
            t0, t1 = min(c[0] for c in counts) - 10, max(c[0] for c in counts) + 10
            edges = np.arange(t0, t1, BIN)
            inc = np.zeros(len(edges) - 1)
            prev = 0
            for t, c in counts:
                i = int((t - t0) / BIN)
                if 0 <= i < len(inc) and c > prev:
                    inc[i] += c - prev
                prev = max(prev, c)
            sh, _ = np.histogram(shots, edges)
            sh = sh - sh.mean()
            inc = inc - inc.mean()
            for k, lag in enumerate(lags):
                s = int(round(lag / BIN))
                a = sh[: len(sh) - s] if s >= 0 else sh[-s:]
                b = inc[s:] if s >= 0 else inc[: len(inc) + s]
                acc[k] += float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            n += 1
    if not n:
        print("no data")
        return
    acc /= n
    best = lags[int(np.argmax(acc))]
    print(f"{n} alliance-matches; peak correlation {acc.max():.3f} at lag {best:.2f} s")
    for lag, c in zip(lags, acc):
        if -0.5 <= lag <= 7:
            print(f"  {lag:5.2f}  {'#' * int(max(0, c) * 200)} {c:.3f}")


if __name__ == "__main__":
    main()
