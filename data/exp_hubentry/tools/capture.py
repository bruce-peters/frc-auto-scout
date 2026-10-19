"""A/B step 1: analyse each scouted Chezy qual in memory once with today's defaults (output/ is not written),
recording every attribution.split_increments call (increments, TeamSignals, OPR, whether the learned model was
on). Variants of the split are then re-run offline on exactly these inputs (ab.py), which is equivalent to
monkeypatching split_increments inside pipeline.analyse but runs in seconds per arm.

Writes data/exp_hubentry/ab/base/<match>/result.json and data/exp_hubentry/capture/<match>.pkl.
Modelled on data/exp_audit/tools/link_x_ab.py.

  .venv/Scripts/python data/exp_hubentry/tools/capture.py [--matches ...]
"""
import argparse
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from autoscout import attribution, config, pipeline, scouting  # noqa: E402

DST = config.DATA / "exp_hubentry"

ap = argparse.ArgumentParser()
ap.add_argument("--matches", nargs="*", default=None)
ap.add_argument("--force", action="store_true")
a = ap.parse_args()

keys = a.matches or sorted(k for k in scouting.by_match(scouting.load()) if (config.OUTPUT / k / "scan.pkl").exists())
print(len(keys), "scouted matches", flush=True)
orig = attribution.split_increments
calls: list = []


def recorder(increments, teams, to_match_time, log=None, opr=None):
    calls.append({"inc": list(increments), "teams": teams, "opr": opr, "model": attribution.ATTR_MODEL,
                  "logged": log is not None})
    return orig(increments, teams, to_match_time, log=log, opr=opr)


attribution.split_increments = recorder
(DST / "capture").mkdir(parents=True, exist_ok=True)
tic = time.time()
for k in keys:
    cp = DST / "capture" / f"{k}.pkl"
    if cp.exists() and not a.force:
        continue
    with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    pipeline.load_dino(md, config.OUTPUT / k)
    calls.clear()
    res = pipeline.analyse(md, pipeline.Options())
    od = DST / "ab" / "base" / k
    od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(json.dumps(res, default=pipeline._json_default))
    tl = md.timeline
    cp.write_bytes(pickle.dumps({"key": k, "calls": list(calls), "red": md.info.red, "blue": md.info.blue,
                                 "auto_start": tl.auto_start, "teleop_start": tl.teleop_start}))
    del md, res
    gc.collect()
    print(k, len(calls), f"{time.time() - tic:.0f}s", flush=True)
attribution.split_increments = orig
