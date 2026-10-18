"""Log every box-linked launch with its box candidates and its hidden teammates (arc_hook.HiddenLogger), on the
54 scored Chezy quals, once with today's defaults (data/exp_arcs/log_hidden/) and once with SPLIT_TRACKS and
GATE_TRACE on (data/exp_arcs/log_hidden_split/). In memory; output/ is not written.

  .venv/Scripts/python -u data/exp_arcs/tools/hidden_log.py
"""
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import config, pipeline  # noqa: E402
import arc_hook  # noqa: E402
from score_scouting import score  # noqa: E402

EXP = config.DATA / "exp_arcs"
keys = sorted({r["key"] for r in score(EXP / "ab_split" / "base", keep_rows=True)["rows"]})
arms = {"log_hidden": {}, "log_hidden_split": {"SPLIT_TRACKS": True, "GATE_TRACE": True}}
arc_hook.build()
print(len(keys), "matches", flush=True)
tic = time.time()
for k in keys:
    with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
        raw = fh.read()
    md = pickle.loads(raw)
    del raw
    pipeline.load_dino(md, config.OUTPUT / k)
    n = {}
    for d, consts in arms.items():
        saved = {c: getattr(pipeline, c) for c in consts}
        for c, v in consts.items():
            setattr(pipeline, c, v)
        lg = arc_hook.HiddenLogger()
        pipeline._ARC_HOOK = lg
        pipeline.analyse_arc(md, pipeline.Options())
        pipeline._ARC_HOOK = None
        for c, v in saved.items():
            setattr(pipeline, c, v)
        (EXP / d).mkdir(exist_ok=True)
        with open(EXP / d / f"{k}.pkl", "wb") as fh:
            pickle.dump(lg.events, fh, protocol=pickle.HIGHEST_PROTOCOL)
        n[d] = (len(lg.events), sum(any(r["src"] == "hidden" for r in e["cands"]) for e in lg.events))
    print(k, f"{time.time() - tic:.0f}s", n, flush=True)
    del md
