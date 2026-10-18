"""Cycle 7: v1 -> v2 regression vs fuel OPR on some events. Which change does it?

For each match of the given events, in memory (OCR patched off: only cached reads are used):
  v1scan_now   : output/ (v1 detections, v1-era OCR cache) analysed with current code (SHOT_GATE on)
  v1scan_nogate: same, SHOT_GATE off
  v2scan_now   : data/rescan_v2 (v2 detections, new OCR) analysed with current code
  v2scan_nogate: same, SHOT_GATE off
plus the stored v1 / v2 result.json. Share corr vs OPR per variant (and per match deltas).
Usage: c7_regress.py EVENT [EVENT ...]
"""
import gc
import json
import pickle
import sys

import numpy as np

from common import AUDIT, DIRS, P, load, match_keys, share_metrics, tba
from autoscout import ocr

events = sys.argv[1:]
ocr.read_numbers = lambda img: []
keys = [k for k in sorted(set(match_keys("v1")) & set(match_keys("v2"))) if any(k.startswith(e + "_") for e in events)]
pm = {v: {} for v in ("v1_stored", "v2_stored", "v1scan_now", "v1scan_nogate", "v2scan_now", "v2scan_nogate")}
for k in keys:
    info = tba.match(k)
    for src in ("v1", "v2"):
        r = load(k, src)
        pm[f"{src}_stored"][k] = {t: r["teams"][str(t)]["fuel"] for t in info.red + info.blue}
        with open(DIRS[src] / k / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        for gate in (True, False):
            P.SHOT_GATE = gate
            res = P.analyse(md, P.Options())
            pm[f"{src}scan_{'now' if gate else 'nogate'}"][k] = {t: res["teams"][t]["fuel"] for t in info.red + info.blue}
        P.SHOT_GATE = True
        del md
        gc.collect()
    print(k, "done", flush=True)
out = {}
from common import scout_metrics
for v, d in pm.items():
    out[v] = {**share_metrics(d), **(scout_metrics(d) or {})}
    print(f"{v:14s} corr {out[v]['corr']:.3f} top {out[v]['top']}/{out[v]['alliances']} MAE {out[v]['mae']:.1f} "
          f"scout {out[v].get('scout_corr', float('nan')):.3f} (n={out[v].get('n')})")
json.dump({"keys": keys, "metrics": out, "fuel": pm}, open(AUDIT / f"c7_regress_{'_'.join(events)}.json", "w"),
          indent=1, default=float)
