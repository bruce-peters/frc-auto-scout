"""Per-crop DINOv2 features with the v5 (DINO-identity) team of each crop -> data/exp_launch/dino/<key>.npz.

crop_id -> detection index from output/<key>/scan.pkl (read-only; scans unchanged since 07:12), detection ->
team / alliance / box from data/exp_attrmodel/v5_snapshot/<key>/dets.json (rows aligned with md.dets).
Keeps box_inner, box_cls, box_mean (float16) for the scouted matches (data/exp_launch/sig/*.pkl).
"""
import gc
import glob
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
import sys  # noqa: E402
sys.path.insert(0, str(ROOT))
EMB, SNAP, OUT = ROOT / "data/exp_dino/emb", ROOT / "data/exp_attrmodel/v5_snapshot", ROOT / "data/exp_launch/dino"
OUT.mkdir(parents=True, exist_ok=True)
VIEWS = ["main", "red_end", "blue_end"]
keys = sys.argv[1:] or sorted(Path(p).stem for p in glob.glob(str(ROOT / "data/exp_launch/sig/*.pkl")))  # default: the scouted matches
for k in keys:
    if (OUT / f"{k}.npz").exists() or not (EMB / f"{k}.npz").exists():
        continue
    E = np.load(EMB / f"{k}.npz")
    rows = json.loads((SNAP / k / "dets.json").read_text())
    with open(ROOT / "output" / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    if len(md.dets) != len(rows):
        print(k, "det count mismatch", len(md.dets), len(rows), flush=True); del md; gc.collect(); continue
    c2d = {d.crop_id: i for i, d in enumerate(md.dets) if getattr(d, "crop_id", None) is not None}
    del md; gc.collect()
    di = np.array([c2d.get(int(c), -1) for c in E["ids"]])
    ok = di >= 0
    team = np.array([(rows[i][7] if rows[i][7] is not None else -1) if i >= 0 else -1 for i in di], np.int32)
    # sanity: crop time / view equal the detection's
    tv_bad = sum(1 for c, i in enumerate(di) if i >= 0 and (abs(rows[i][0] - float(E["t"][c])) > 2e-3 or rows[i][1] != VIEWS[int(E["view"][c])]))
    box = np.array([rows[i][2:6] if i >= 0 else [np.nan] * 4 for i in di], np.float32)
    np.savez(OUT / f"{k}.npz", t=E["t"], view=E["view"], bh=E["bh"], team=team, det=di.astype(np.int32), box=box,
             box_inner=E["box_inner"], box_cls=E["box_cls"], box_mean=E["box_mean"])
    print(k, f"{ok.sum()}/{len(ok)} crops mapped, {int((team >= 0).sum())} with a team, t/view mismatches {tv_bad}", flush=True)
