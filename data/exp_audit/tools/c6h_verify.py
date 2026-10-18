"""Cycle 7h: verify 'no zone gating in auto' on top of the current code (SEG_MAX etc.).

For each output/ match (v2 scans): current pipeline.analyse() in memory (OCR patched to cached reads),
then re-split the counter with zone gating as is vs off in auto (c6e_auto_variants.make). Paired
bootstrap of the total-share corr vs fuel OPR, plus scouting. Writes data/exp_audit/c6h_verify.json.
"""
import gc
import json
import pickle
import sys

from common import A, AUDIT, DIRS, P, TL, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals
from autoscout import ocr

ocr.read_numbers = lambda img: []
from c6e_auto_variants import make, ORIG  # noqa: E402  (that module also runs its own table on import)

src = "v2"
out = {"fuel": {"as_is": {}, "off": {}}, "seg_max": P.SEG_MAX}
for k in match_keys(src):
    with open(DIRS[src] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    res = P.analyse(md, P.Options())
    del md
    gc.collect()
    res["paths"] = {str(t): v for t, v in res["paths"].items()}
    res["teams"] = {str(t): v for t, v in res["teams"].items()}
    info = tba.match(k)
    sig = team_signals(res, info)
    for mode in ("as_is", "off"):
        A.split_increments = make(mode, TL(res))
        out["fuel"][mode][k] = {str(t): f for t, f in resplit(res, info, signals=sig).items()}
        A.split_increments = ORIG
    print(k, "done", flush=True)
for mode, d in out["fuel"].items():
    pm = {k: {int(t): f for t, f in v.items()} for k, v in d.items()}
    print(mode, share_metrics(pm), scout_metrics(pm))
json.dump(out, open(AUDIT / "c6h_verify.json", "w"))
