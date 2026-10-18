"""Parity: vol_signal (production function) vs the dataset's per-jump s_beq values.

  c28_parity.py cache DS.pkl key [key ...]   : vol rows from data/exp_audit/vol_cache (vol_signal run on the scans)
  c28_parity.py scan  DS.pkl SRC key [...]  : vol rows computed now from SRC/<key>/scan.pkl with the team labels of
                                              SRC/<key>/dets.json (e.g. after porting: SRC = output, DS rebuilt from
                                              the same results - labels must match the dataset's)
For every scoreboard jump of every alliance of those matches: s_beq per robot from vol_signal.s_beq vs the
dataset's X[..., 's_beq']; prints max |diff| and the share of jumps that match to 1e-6.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from common import AUDIT, ROOT
import vol_signal as VS

mode, ds = sys.argv[1], sys.argv[2]
src = Path(sys.argv[3]) if mode == "scan" else None
keys = sys.argv[4:] if mode == "scan" else sys.argv[3:]
with open(ds, "rb") as fh:
    d = pickle.load(fh)
fi = d["feats"].index("s_beq")
MS = float(os.environ.get("VOL_MIN_SPEED", VS.MIN_SPEED))  # must match the dataset's VOL_MIN_SPEED (ds_vol.pkl: 0)
for k in keys:
    if mode == "cache":
        z = np.load(AUDIT / "vol_cache" / f"{k}.npz")
        vol = {int(t): z[t][z[t][:, 2] >= MS][:, :2] for t in z.files}
    else:
        rows = json.loads((src / k / "dets.json").read_text())
        with open(src / k / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        vol = VS.ball_equivalents(md.tracks_all, md.dets, [r[7] for r in rows], md.cams, md.prod, min_speed=MS)
        del md
    diffs, n = [], 0
    for rec in d["recs"]:
        if rec["key"] != k or not rec.get("overlay") or not len(rec["n"]):
            continue
        for ri, team in enumerate(rec["teams"]):
            mine = np.array([VS.s_beq(vol.get(team), t) for t in rec["t"]])
            ref = rec["X"][:, ri, fi]
            diffs.append(np.abs(mine - ref)); n += len(ref)
    dd = np.concatenate(diffs) if diffs else np.zeros(0)
    print(f"{k}: {n} robot-jumps, max |diff| {dd.max() if len(dd) else float('nan'):.2e}, "
          f"match (1e-6) {100 * np.mean(dd < 1e-6) if len(dd) else float('nan'):.1f} %")
