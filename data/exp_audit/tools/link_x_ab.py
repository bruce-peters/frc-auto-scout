"""A/B of pipeline.LINK_X_TIE (launch-to-box tie-break by horizontal position) on the scouted Chezy quals.
Each scan is analysed in memory once per variant (output/ is not written); result.json goes to
data/exp_linkx/<variant>/<match>/ and is scored with scripts/score_scouting.score, plus the per-team burst
check of data/exp_audit/tools/team_volleys.py for 254.

  .venv/Scripts/python data/exp_audit/tools/link_x_ab.py [--variants 0,0.5,1,2]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from autoscout import config, pipeline, scouting  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default="0,0.5,1,2")
ap.add_argument("--cross", action="store_true", help="LINK_X_CROSS: horizontal position where the arc crossed the box top")
ap.add_argument("--matches", nargs="*", default=None)
ap.add_argument("--set", nargs="*", default=[], help="pipeline module constants for every arm, NAME=value")
ap.add_argument("--arms", default=None, help="instead of LINK_X_TIE variants: ';'-separated arms of NAME=value,NAME=value")
ap.add_argument("--dst", default=None)
a = ap.parse_args()
def _val(x):
    return {"True": True, "False": False}.get(x, float(x) if x.replace(".", "", 1).lstrip("-").isdigit() else x)


def _owner(k):  # "NAME" -> pipeline, "attribution.NAME" -> pipeline.attribution
    mod, _, name = k.rpartition(".")
    return (getattr(pipeline, mod) if mod else pipeline), name


for kv in a.set:
    k_, v_ = kv.split("=")
    setattr(*_owner(k_), _val(v_))
pipeline.LINK_X_CROSS = a.cross or pipeline.LINK_X_CROSS
if a.arms:  # named arms: each sets some pipeline constants
    arms = [dict(kv.split("=") for kv in arm.split(",") if kv) for arm in a.arms.split(";")]
    variants = [",".join(f"{k}={v}" for k, v in arm.items()) or "base" for arm in arms]
else:
    arms = None
    variants = [float(x) for x in a.variants.split(",")]
dst = Path(a.dst) if a.dst else config.DATA / ("exp_linkx_cross" if a.cross else "exp_linkx")
keys = a.matches or sorted(k for k in scouting.by_match(scouting.load()) if (config.OUTPUT / k / "scan.pkl").exists())
print(len(keys), "scouted matches", flush=True)
tic = time.time()
for k in keys:
    with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    pipeline.load_dino(md, config.OUTPUT / k)
    pipeline.load_fuel60(md)  # 60 Hz flight tracks when pipeline.FUEL_60HZ and the match has a 60 Hz blob log
    for vi, x in enumerate(variants):
        saved = {}
        if arms is not None:
            for k_, v_ in arms[vi].items():
                saved[k_] = getattr(*_owner(k_))
                setattr(*_owner(k_), _val(v_))
            name = x
        else:
            pipeline.LINK_X_TIE = x
            name = f"x{x:g}"
        res = pipeline.analyse(md, pipeline.Options())
        res["fuel_hz"] = md.fuel_hz
        for k_, v_ in saved.items():
            setattr(*_owner(k_), v_)
        od = dst / name / k
        od.mkdir(parents=True, exist_ok=True)
        (od / "result.json").write_text(json.dumps(res, default=pipeline._json_default))
    print(k, f"{time.time() - tic:.0f}s", flush=True)

from score_scouting import score  # noqa: E402

for x in variants:
    s = score(dst / (x if arms is not None else f"x{x:g}"))
    v, w = s["video"], s["vonly"]
    print(f"{x}: best estimate MAE {v['mae']:.1f} corr {v['corr']:.3f} top {100 * s['top_scorer']['video']:.0f} % | "
          f"video only MAE {w['mae']:.1f} corr {w['corr']:.3f} top {100 * s['top_scorer']['vonly']:.0f} % | "
          f"burst {100 * s['burst_overlap']['video']:.1f} %", flush=True)
