"""dets.json for link_x_ab.py arms (the harness writes result.json only; build_ds.py also reads dets.json).

Arms that leave identity alone get output/<match>/dets.json; arms that change it (IDENT_FAR_COST) are re-analysed.

  .venv/Scripts/python data/exp_audit/tools/arm_dets.py DST_DIR
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, pipeline  # noqa: E402

dst = Path(sys.argv[1])
keys = (ROOT / "data" / "exp_qafix_keys.txt").read_text().split()
arms = [p for p in dst.iterdir() if p.is_dir()]
recompute = [p for p in arms if "IDENT_FAR_COST=20" in p.name]
assert all("IDENT_FAR_COST=20" in p.name or "IDENT_FAR_COST" not in p.name for p in arms)
for p in arms:
    if p in recompute:
        continue
    for k in keys:
        (p / k).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(config.OUTPUT / k / "dets.json", p / k / "dets.json")
print("copied", [p.name for p in arms if p not in recompute], flush=True)
for k in keys:  # identity depends on IDENT_FAR_COST only: one re-analysis per match serves every such arm
    md = pickle.load(open(config.OUTPUT / k / "scan.pkl", "rb"))
    pipeline.load_dino(md, config.OUTPUT / k)
    saved = pipeline.IDENT_FAR_COST
    pipeline.IDENT_FAR_COST = 20.0
    pipeline.analyse(md, pipeline.Options())
    pipeline.IDENT_FAR_COST = saved
    for p in recompute:
        (p / k).mkdir(parents=True, exist_ok=True)
        (p / k / "dets.json").write_text(json.dumps(md.det_rows, default=pipeline._json_default))
    print(k, flush=True)
print("done", flush=True)
