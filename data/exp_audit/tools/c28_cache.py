"""Cycle 28: read every output/ scan ONCE (before the DINO reanalysis rewrites them) and cache, per match,
vol_signal.ball_equivalents(..., with_speed=True) with the v4_snapshot dets.json team labels -> data/exp_audit/vol_cache/<key>.npz."""
import gc, json, os, pickle, sys
from pathlib import Path
import numpy as np
from common import AUDIT, ROOT
import vol_signal as VS
# env overrides for a re-run on new results (e.g. after the DINO reanalysis: VOL_LABELS=output VOL_CACHE=.../vol_cache_dino)
SNAP = Path(os.environ.get("VOL_LABELS", ROOT / "data" / "exp_attrmodel" / "v4_snapshot"))  # dets.json team labels
OUT = Path(os.environ.get("VOL_CACHE", AUDIT / "vol_cache")); OUT.mkdir(exist_ok=True)
keys = sorted(p.parent.name for p in SNAP.glob("2026*/dets.json"))
for k in keys:
    if (OUT / f"{k}.npz").exists():
        continue
    rows = json.loads((SNAP / k / "dets.json").read_text())
    with open(ROOT / "output" / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    if len(md.dets) != len(rows):
        print(k, "dets mismatch", len(md.dets), len(rows), flush=True); del md; gc.collect(); continue
    tmis = sum(1 for d, r in zip(md.dets, rows) if abs(round(d.t, 3) - r[0]) > 1e-3 or d.view != r[1])
    vol = VS.ball_equivalents(md.tracks_all, md.dets, [r[7] for r in rows], md.cams, md.prod, with_speed=True)
    np.savez(OUT / f"{k}.npz", **{str(t): v for t, v in vol.items()})
    print(k, "rows", sum(len(v) for v in vol.values()), "det order mismatches", tmis, flush=True)
    del md; gc.collect()
