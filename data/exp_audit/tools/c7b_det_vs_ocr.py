"""Cycle 7b: split the v1 -> v2 change into detector and bumper-OCR parts (30-match A/B subset).

Same current analyse() for all, OCR patched off (only the reads cached in each scan are used):
  A  data/exp_robots_v2/v1/<m>/scan.pkl : v1 detector, old OCR reads (64 px prep, 12/seg, 700 budget)
  B  data/exp_robots_v2/v2/<m>/scan.pkl : v2 detector, old OCR reads
  C  data/rescan_v2/<m>/scan.pkl        : v2 detector, new OCR reads (128 px prep, more reads)
Metrics: share corr vs OPR (all + per event), held-out identity (ocr_holdout 0/1), scouting.
Usage: c7b_det_vs_ocr.py
"""
import gc
import json
import pickle
from collections import defaultdict

import numpy as np

from common import AUDIT, DIRS, P, ROOT, scout_metrics, share_metrics, tba
from autoscout import ocr

ocr.read_numbers = lambda img: []
SRC = {"A_v1det_oldocr": ROOT / "data/exp_robots_v2/v1", "B_v2det_oldocr": ROOT / "data/exp_robots_v2/v2",
       "C_v2det_newocr": ROOT / "data/rescan_v2"}
keys = (ROOT / "data/exp_robots_v2/subset.txt").read_text().split()
keys = [k for k in keys if all((d / k / "scan.pkl").exists() for d in SRC.values())]
pm = {v: {} for v in SRC}
held = {v: [0.0, 0.0] for v in SRC}
for k in keys:
    info = tba.match(k)
    for v, d in SRC.items():
        with open(d / k / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        res = P.analyse(md, P.Options())
        pm[v][k] = {t: res["teams"][t]["fuel"] for t in info.red + info.blue}
        for hk in (0, 1):
            rh = P.analyse(md, P.Options(ocr_holdout=hk))
            for al in ("red", "blue"):
                tm = info.teams(al)
                for s in rh["identity"][al].values():
                    if s["team"] is not None:
                        held[v][0] += s["held"][tm.index(s["team"])]
                        held[v][1] += sum(s["held"])
        del md
        gc.collect()
    print(k, "done", flush=True)
out = {}
for v in SRC:
    m = share_metrics(pm[v]); s = scout_metrics(pm[v]) or {}
    ev = defaultdict(dict)
    for k, f in pm[v].items():
        ev[tba.match(k).event_key][k] = f
    out[v] = {"heldout": held[v][0] / max(held[v][1], 1e-9), **m, **s,
              "by_event": {e: round(share_metrics(d)["corr"], 3) for e, d in sorted(ev.items())}}
    print(v, json.dumps(out[v]))
json.dump({"keys": keys, "metrics": out, "fuel": pm}, open(AUDIT / "c7b_det_vs_ocr.json", "w"), indent=1, default=float)
