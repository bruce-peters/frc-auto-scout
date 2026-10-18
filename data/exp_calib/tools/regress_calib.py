"""No-regression check of the calibration path: re-run pipeline.calibrate from the video for each
match, with the old call (no event) and with the new event path (event prior + recovery), and compare
every view that calibrates with the camera stored in the scan (K, distortion, pose times, rvecs, tvecs).
The event-prior cache is not written (update_calib_prior is recorded only). Read-only on the scans.
Usage: regress_calib.py SCAN_DIR MATCH [MATCH ...]
"""
import gc
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import config, field as F, pipeline as P  # noqa: E402
from autoscout.video import Video  # noqa: E402

RECORDED = []
F.update_calib_prior = lambda event, view, match, cam: RECORDED.append((event, view, match, cam.rms)) or True


def diff(a, b) -> float:
    """Largest absolute difference between two cameras' parameters (inf if their pose times differ)."""
    if a is None or b is None:
        return float("inf") if (a is None) != (b is None) else 0.0
    ta, tb = list(getattr(a, "times", [0])), list(getattr(b, "times", [0]))
    if len(ta) != len(tb) or not np.allclose(ta, tb):
        return float("inf")
    parts = [(a.K, b.K), (a.dist, b.dist)]
    if hasattr(a, "rvecs"):
        parts += [(a.rvecs, b.rvecs), (a.tvecs, b.tvecs)]
    return max(float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float)))) for x, y in parts)


src = Path(sys.argv[1])
out = {}
for key in sys.argv[2:]:
    with open(src / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    stored, tl, prod, ev = dict(md.cams), md.timeline, md.prod, md.info.event_key
    del md
    gc.collect()
    vid = Video(config.VIDEOS / f"{key}.mp4")
    tele = tl.teleop_start or (tl.auto_start + P.FIRST_TELEOP_GAP)
    t0, t1 = tl.auto_start - 1, tele + config.TELEOP_LEN + 2
    old = P.calibrate(vid, prod, t0, t1)
    fb = {}
    new = P.calibrate(vid, prod, t0, t1, event=ev, match=key, fallbacks=fb)
    vid.close()
    row = {}
    for v in sorted(set(stored) | set(old) | set(new)):
        row[v] = {"stored": v in stored, "old": v in old, "new": v in new,
                  "old_vs_new": diff(old.get(v), new.get(v)), "new_vs_stored": diff(new.get(v), stored.get(v))}
    row["_recovery_candidates"] = {v: [h for h, _ in c] for v, c in fb.items()}
    out[key] = row
    print(key, json.dumps(row, default=float), flush=True)
    (ROOT / "data" / "exp_calib" / "regress_calib.json").write_text(json.dumps(out, indent=1, default=float),
                                                                   encoding="utf-8")
print("prior updates recorded (not written):", len(RECORDED))
