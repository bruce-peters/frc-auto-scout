"""Leave-one-out check of the fixed-intrinsics fallback against end views that calibrate on their own.

For each match / end view with its own calibration: pretend it failed, rebuild it with
field.calibrate_view_fixed from the event prior (this match excluded), using all tags, only the two
largest, or only the largest tag per frame (emulating the failing matches). Compare the robot positions
it gives with those from the view's own calibration (same boxes, placed as analyse() does), and run the
cross-view gate on both. Read-only on the scans. Writes data/exp_calib/validate_loo.json.
Usage: validate_loo.py SCAN_DIR MATCH [MATCH ...]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

from autoscout import config, field as F, pipeline as P, robots, views  # noqa: E402
from autoscout.video import Video  # noqa: E402
from gate_baseline_lib import null_error  # noqa: E402

args = [x for x in sys.argv[1:] if not x.startswith("--out=")]
OUT = next((x[6:] for x in sys.argv[1:] if x.startswith("--out=")), "validate_loo.json")
src = Path(args[0])
keys = args[1:]
_detect = F._detect_masked
KEEP = [None]


def detect_limited(img, masks):
    d = _detect(img, masks)
    if KEEP[0] is None or len(d) <= KEEP[0]:
        return d
    size = {i: np.ptp(c[:, 0]) * np.ptp(c[:, 1]) for i, c in d.items()}
    return {i: d[i] for i in sorted(d, key=lambda i: -size[i])[:KEEP[0]]}


F._detect_masked = detect_limited


def positions(md, vname, cam, limit):
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    out = {}
    for k, d in enumerate(md.dets):
        if d.view != vname:
            continue
        ox, oy = vl[vname].offset(1920, 1080)
        c = cam.at(d.t)
        xy = c.backproject(np.array([[(d.box[0] + d.box[2]) / 2 - ox, d.box[3] - oy]]), z=0.03)[0]
        if not np.all(np.isfinite(xy)):
            continue
        dv = xy - c.center_world()[:2]
        rng = float(np.hypot(*dv))
        xy = xy + P.ROBOT_HALF_DEPTH * dv / max(rng, 1e-6)
        if robots.in_field(xy) and (limit is None or rng <= limit):
            out[k] = xy
    return out


path = ROOT / "data" / "exp_calib" / OUT
allres = json.loads(path.read_text()) if path.exists() else {}
for key in keys:
    with open(src / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    tl = md.timeline
    tele = tl.teleop_start or (tl.auto_start + P.FIRST_TELEOP_GAP)
    times = [float(t) for t in np.arange(tl.auto_start - 1, tele + config.TELEOP_LEN + 2, 4.0)]
    vid = Video(config.VIDEOS / f"{key}.mp4")
    fr = [(t, vid.frame_at(t)) for t in times]
    fr = [(t, f) for t, f in fr if f is not None]
    times = [t for t, _ in fr]
    tags = F.load_tags()
    for view in views.PRODUCTIONS[md.prod]:
        own = md.cams.get(view.name)
        if view.name == "main" or own is None:
            continue
        crops = [view.crop(f) for _, f in fr]
        masks = view.mask_boxes(vid.width, vid.height)
        prior = F.load_calib_prior(md.info.event_key, view.name, crops[0].shape[1::-1], exclude=key)
        g_own = P.cross_view_error(md, view.name, own)
        p_own = positions(md, view.name, own, P.END_VIEW_RANGE_COV)
        row = {"own": {"rms": own.rms, "f": own.K[0][0], "gate": g_own["median"], "n": g_own["n"],
                       "null": null_error(md, view.name, own), "centre": F.track_centre(own).round(3).tolist()},
               "prior_f": prior["f"], "prior_centre": np.round(prior["centre"], 3).tolist()}
        for name, keep in (("all", None), ("2tags", 2), ("1tag", 1)):
            KEEP[0] = keep
            diag = {}
            fb = F.calibrate_view_fixed(crops, times, tags, masks, prior, diag=diag)
            KEEP[0] = None
            if fb is None:
                row[name] = {"poses": 0}
                print(f"{key} {view.name} {name}: no fallback", flush=True)
                continue
            p_fb = positions(md, view.name, fb, None)
            common = [k for k in p_own if k in p_fb]
            dd = np.array([np.hypot(*(p_own[k] - p_fb[k])) for k in common])
            g = P.cross_view_error(md, view.name, fb)
            row[name] = {"poses": len(fb.times), "rms": fb.rms, "centre_from": diag.get("centre_from"),
                         "centre": diag.get("centre"), "pos_diff_median": float(np.median(dd)) if len(dd) else None,
                         "pos_diff_q90": float(np.quantile(dd, 0.9)) if len(dd) else None, "n": len(dd),
                         "gate": g["median"], "gate_n": g["n"],
                         "gate_pass": bool(g["n"] >= P.CALIB_GATE_MIN_N and g["median"] < P.CALIB_GATE_M)}
            r = row[name]
            print(f"{key} {view.name} {name:5s}: poses {r['poses']:2d} rms {r['rms']:.2f} centre {r['centre_from']}; "
                  f"vs own calibration: robot positions differ median {r['pos_diff_median']:.2f} m q90 "
                  f"{r['pos_diff_q90']:.2f} (n {r['n']}); gate {r['gate']:.2f} (own {g_own['median']:.2f}) "
                  f"{'PASS' if r['gate_pass'] else 'fail'}", flush=True)
        allres[f"{key}|{view.name}"] = row
    vid.close()
    del md
    gc.collect()
    path.write_text(json.dumps(allres, indent=1, default=float), encoding="utf-8")
