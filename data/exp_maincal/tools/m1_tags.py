"""Main-camera tag sightings per sampled frame (every 4 s, as pipeline.calibrate) for a few matches.

Is the Chezy main camera static, panning or zooming? Per frame: tag ids, centre pixel and width of each
tag; per match: per-tag spread of the centre over the match and the width range. Also the stored camera.
Usage: m1_tags.py match...   (reads output/<match>/result.json + scan.pkl, data/videos/<match>.mp4)
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

from autoscout import config, field as F, pipeline as P, views  # noqa: E402
from autoscout.video import Video  # noqa: E402

OUT = ROOT / "data" / "exp_maincal"
res_all = {}
for key in sys.argv[1:]:
    r = json.loads((ROOT / "output" / key / "result.json").read_text())
    a0 = r["timeline"]["auto_start"] or P.FIRST_AUTO_START
    tele = r["timeline"]["teleop_start"] or a0 + P.FIRST_TELEOP_GAP
    times = [float(t) for t in np.arange(a0 - 1, tele + config.TELEOP_LEN + 2, 4.0)]
    v = Video(config.VIDEOS / f"{key}.mp4")
    prod = "chezy" if key.startswith("2026cc") else "first"
    view = next(x for x in views.PRODUCTIONS[prod] if x.name == "main")
    masks = view.mask_boxes(v.width, v.height)
    rows = []
    for t in times:
        f = v.frame_at(t)
        if f is None:
            continue
        d = F._detect_masked(view.crop(f), masks)
        rows.append((t, {i: c.tolist() for i, c in d.items()}))
    v.close()
    with open(ROOT / "output" / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cam = md.cams.get("main")
    camd = None
    if cam is not None:
        cents = np.array([cam.at(t).center_world() for t in getattr(cam, "times", [0.0])])
        camd = {"f": cam.K[0][0], "k1": cam.dist[0], "rms": cam.rms, "poses": len(getattr(cam, "times", [0])),
                "times": list(getattr(cam, "times", [])), "centres": cents.round(3).tolist(),
                "rvecs": [list(map(float, x)) for x in (cam.rvecs if hasattr(cam, "rvecs") else [cam.rvec])]}
    del md
    gc.collect()
    res_all[key] = {"frames": rows, "cam": camd}
    print(f"== {key}: {len(rows)} frames, cam f {camd and round(camd['f'])} k1 {camd and round(camd['k1'], 3)} "
          f"poses {camd and camd['poses']}")
    ids = sorted({i for _, d in rows for i in d})
    for i in ids:
        cs = np.array([np.mean(d[i], axis=0) for _, d in rows if i in d])
        ws = np.array([np.linalg.norm(np.array(d[i])[0] - np.array(d[i])[1]) for _, d in rows if i in d])
        hs = np.array([np.linalg.norm(np.array(d[i])[0] - np.array(d[i])[3]) for _, d in rows if i in d])
        print(f"   tag {i:2d}: seen {len(cs):2d}  centre med {np.median(cs, 0).round(1)}  std {cs.std(0).round(2)}  "
              f"range u {np.ptp(cs[:, 0]):.1f} v {np.ptp(cs[:, 1]):.1f}  width {np.median(ws):.2f} ({ws.min():.1f}-{ws.max():.1f})"
              f"  height {np.median(hs):.2f}")
    if camd:
        c = np.array(camd["centres"])
        print(f"   cam centres median {np.median(c, 0).round(2)}  std {c.std(0).round(2)}")
json.dump(res_all, open(OUT / "m1_tags.json", "w"), indent=0)
