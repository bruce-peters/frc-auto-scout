"""One light pass over output/<match>/scan.pkl (read-only, one scan in memory at a time): per view the flight
track table of autoscout.hubentry.track_ends, plus the points of every track that ends in or near a hub
opening. Cached to data/exp_hubentry/cache/<match>.pkl.

  .venv/Scripts/python data/exp_hubentry/tools/extract.py [--matches ...] [--force]
"""
import argparse
import gc
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import config, hubentry as H  # noqa: E402

CACHE = config.DATA / "exp_hubentry" / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = a.matches or sorted(p.parent.name for p in config.OUTPUT.glob("*/scan.pkl"))
    tic = time.time()
    for k in keys:
        dst = CACHE / f"{k}.pkl"
        if dst.exists() and not a.force:
            continue
        with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        out = {"key": k, "prod": md.prod, "views": {}, "near_pts": {}, "opening": {}}
        for v, trs in md.tracks_all.items():
            cam = md.cams.get(v)
            T = H.track_ends(trs, cam)
            out["views"][v] = T.astype(np.float32)
            if cam is not None:
                out["opening"][v] = {al: H.opening_frames(cam, al) for al in ("red", "blue")}
            # points of tracks ending near either opening (for linking to launches)
            kept = [tr for tr in trs if len(tr.pts) >= H.MIN_PTS]
            near = np.zeros(len(T), bool)
            for al in ("red", "blue"):
                with np.errstate(invalid="ignore"):
                    near |= (np.abs(T[:, H.C[f"{al}_du"]]) < 2.5) & (np.abs(T[:, H.C[f"{al}_dv"]]) < 3.0)
            out["near_pts"][v] = {int(i): np.asarray(kept[i].pts, dtype=np.float32) for i in np.flatnonzero(near)}
        dst.write_bytes(pickle.dumps(out))
        del md, out
        gc.collect()
        print(k, f"{time.time() - tic:.0f}s", flush=True)


if __name__ == "__main__":
    main()
