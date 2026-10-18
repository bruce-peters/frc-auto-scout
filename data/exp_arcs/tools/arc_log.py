"""Pass 1: analyse each scan in memory with the logging hook (output/ is not written).

Per match: data/exp_arcs/log/<match>.pkl (every launch with >= 2 candidate boxes within the link distance, see
arc_hook.Logger) and data/exp_arcs/base/<match>/result.json (the unchanged analysis = the A/B baseline arm).
Matches: the c24 truth-set matches and the scouted Chezy quals that have a scan.

  .venv/Scripts/python -u data/exp_arcs/tools/arc_log.py [--parity] [--matches ...]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import config, pipeline, scouting  # noqa: E402
import arc_hook  # noqa: E402

EXP = config.DATA / "exp_arcs"
ap = argparse.ArgumentParser()
ap.add_argument("--matches", nargs="*", default=None)
ap.add_argument("--parity", action="store_true", help="also run the original pipeline.analyse on the first match and compare")
a = ap.parse_args()

truth = json.loads((config.DATA / "exp_audit" / "c24_truth_cases.json").read_text())
keys = a.matches or sorted({c["key"] for c in truth} | set(scouting.by_match(scouting.load())))
keys = [k for k in keys if (config.OUTPUT / k / "scan.pkl").exists()]
print(len(keys), "matches", flush=True)
arc_hook.build()
(EXP / "log").mkdir(parents=True, exist_ok=True)
tic = time.time()
for n, k in enumerate(keys):
    with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
        raw = fh.read()  # read and close at once: the main session may swap scan.pkl in place
    md = pickle.loads(raw)
    del raw
    pipeline.load_dino(md, config.OUTPUT / k)
    if a.parity and n == 0:
        pipeline._ARC_HOOK = None
        ref = json.loads(json.dumps(pipeline.analyse(md, pipeline.Options()), default=pipeline._json_default))
    log = arc_hook.Logger()
    pipeline._ARC_HOOK = log
    res = pipeline.analyse_arc(md, pipeline.Options())
    pipeline._ARC_HOOK = None
    js = json.dumps(res, default=pipeline._json_default)
    if a.parity and n == 0:
        same = {kk: vv for kk, vv in json.loads(js).items() if kk != "runtime_s"} == {kk: vv for kk, vv in ref.items() if kk != "runtime_s"}
        print("parity analyse_arc(Logger) vs pipeline.analyse:", "identical" if same else "DIFFERENT", flush=True)
    od = EXP / "base" / k
    od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(js)
    with open(EXP / "log" / f"{k}.pkl", "wb") as fh:
        pickle.dump(log.events, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(k, len(log.events), "ambiguous launches", f"{time.time() - tic:.0f}s", flush=True)
    del md, log, res
