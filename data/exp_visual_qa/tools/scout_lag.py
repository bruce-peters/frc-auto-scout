"""Clock check of every scout row: cross-correlate its taps (shifted -0.8 s to launch time) with the alliance's
counter increments (shifted -2.4 s to launch time; independent of the credit split) over lags -20..+20 s. A row
whose best lag is far from 0 has a clock offset (or logs in late batches). Also: the share of a row's balls entered
as batch taps (>= 5 identical timestamps).

  .venv/Scripts/python data/exp_visual_qa/tools/scout_lag.py [--out data/exp_visual_qa/scout_lag.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import scouting, tba  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "data/exp_visual_qa/scout_lag.json"))
a = ap.parse_args()
C = ROOT / "data/exp_visual_qa/cache"
bm = scouting.by_match(scouting.load())
B = 0.5
edges = np.arange(0, 200 + B, B)
k = np.ones(4) / 4  # 2 s box
lags = np.arange(-40, 41)  # bins
out = []
for d in sorted(C.iterdir()):
    if not (d / "result.json").exists() or d.name not in bm:
        continue
    r = json.loads((d / "result.json").read_text())
    a0 = r["timeline"].get("auto_start") or 6.13
    info = tba.match(d.name)
    for al in ("red", "blue"):
        inc = np.zeros(len(edges) - 1)
        prev = 0
        for vt, c in r["timeline"]["counts"].get(al, []):
            if c > prev:
                j = np.searchsorted(edges, vt - 2.4) - 1
                if 0 <= j < len(inc):
                    inc[j] += c - prev
                prev = c
        y = np.convolve(inc, k, "same")
        for t in info.teams(al):
            for i, row in enumerate(bm[d.name].get(t, [])):
                if row.n < 10:
                    continue
                x = np.convolve(np.histogram(row.shots + a0 - 0.8, edges)[0].astype(float), k, "same")
                cc = []
                for L in lags:  # x shifted by L bins: x(t - L)
                    xs = np.roll(x, L)
                    if L > 0:
                        xs[:L] = 0
                    elif L < 0:
                        xs[L:] = 0
                    cc.append(float((xs * y).sum() / (np.linalg.norm(xs) * np.linalg.norm(y) + 1e-9)))
                cc = np.array(cc)
                best = int(lags[np.argmax(cc)])
                _, cnt = np.unique(row.shots, return_counts=True)
                batch = float(cnt[cnt >= 5].sum() / row.n)
                out.append({"match": d.name, "team": t, "alliance": al, "row": i, "n": int(row.n),
                            "valid": row.time_valid, "best_lag_s": best * B, "cc0": round(float(cc[lags == 0][0]), 3),
                            "cc_best": round(float(cc.max()), 3), "batch_share": round(batch, 2)})
Path(a.out).write_text(json.dumps(out, indent=0))
v = [o for o in out if o["valid"]]
L = np.array([o["best_lag_s"] for o in v])
print(len(out), "scout rows with >= 10 shots,", len(v), "time-valid")
print("best lag (s; + = taps late vs counter): |lag|<=1.5:", int((np.abs(L) <= 1.5).sum()), " 1.5-5:",
      int(((np.abs(L) > 1.5) & (np.abs(L) <= 5)).sum()), " >5:", int((np.abs(L) > 5).sum()))
print("median lag", np.median(L), " mean batch share", round(float(np.mean([o["batch_share"] for o in v])), 2))
for o in sorted(v, key=lambda o: -abs(o["best_lag_s"]))[:25]:
    print(o)
