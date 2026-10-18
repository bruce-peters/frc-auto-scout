"""prod_signals wrappers on pipeline objects (md from scan.pkl, det_seg / team_of_seg rebuilt from the dets.json sid and
team columns, i.e. what analyse holds) vs the series build_ds.py stores (dets.json path), per match.
Differences come only from Detection.fill at full precision vs dets.json's 3 decimals.

  wrapper_parity.py SRC_DIR key [key ...]     (SRC_DIR: the results dir build_ds read, e.g. v5_snapshot)
"""
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "data" / "exp_launch"))
sys.path.insert(0, str(ROOT / "data" / "exp_attrmodel" / "tools")); sys.path.insert(0, str(ROOT / "data" / "exp_audit" / "tools"))
import build_ds as BD  # noqa: E402
import prod_signals as PS  # noqa: E402
from common import TL  # noqa: E402

src = Path(sys.argv[1])
for k in sys.argv[2:]:
    r = json.loads((src / k / "result.json").read_text())
    rows = json.loads((src / k / "dets.json").read_text())
    tl = TL(r)
    ref, _ = BD.shoot_series(k, tl, rows)
    with open(ROOT / "output" / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    assert len(md.dets) == len(rows)
    det_seg = {id(d): rows[i][8] for i, d in enumerate(md.dets) if rows[i][8] is not None}
    team_of_seg = {rows[i][8]: rows[i][7] for i in range(len(rows)) if rows[i][8] is not None}
    grid = np.arange(tl.t0(), tl.t1(), PS.GRID_STEP)
    dc = PS.dinoc_series(md, ROOT / "data" / "exp_dino" / "emb" / f"{k}.npz", ROOT / "models" / "shooting_ridge.npz",
                         det_seg, team_of_seg, grid=grid, exclude_key=k)
    dr = PS.drain_series(md, det_seg, team_of_seg, grid=grid)
    del md; gc.collect()
    md_d = max(np.abs(dc[t][1] - ref[t][0]).max() for t in dc if ref.get(t, (None,))[0] is not None)
    md_r = max(np.abs(dr[t][1] - ref[t][1]).max() for t in dr if t in ref)
    fr = [(PS.jump_feature(grid, dr[t][1], x, "window"), PS.jump_feature(grid, ref[t][1], x, "window"))
          for t in dr if t in ref for x, _ in A_inc] if (A_inc := [(x, 0) for x in grid[::4]]) else []
    fr = np.array(fr)
    print(f"{k}: dinoc series max |diff| {md_d:.2e} over {len(dc)} robots; drain series max |diff| {md_r:.2e}, "
          f"drain window feature max |diff| {np.abs(fr[:, 0] - fr[:, 1]).max():.2e} (mean {fr[:, 1].mean():.4f})", flush=True)
