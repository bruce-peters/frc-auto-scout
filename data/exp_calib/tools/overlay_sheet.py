"""Contact sheet: field lines, hub openings and the visible AprilTags' outlines projected with each
recovered end camera onto frames of that view (start, middle, late), for an eyeball check.
Writes data/exp_calib/img/recovered_views.jpg. Read-only on the scans.
Usage: overlay_sheet.py SCAN_DIR MATCH:VIEW [MATCH:VIEW ...]
"""
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from autoscout import config, field as F, views  # noqa: E402
from autoscout.video import Video  # noqa: E402

src = Path(sys.argv[1])
tags = F.load_tags()
rows = []
for item in sys.argv[2:]:
    key, vname = item.split(":")
    with open(src / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cam = md.cams[vname]
    view = next(v for v in views.PRODUCTIONS[md.prod] if v.name == vname)
    vid = Video(config.VIDEOS / f"{key}.mp4")
    ts = np.asarray(cam.times)
    tiles = []
    for t in (ts[0] + 2, float(np.median(ts)) + 2, ts[-1] - 6):
        img = view.crop(vid.frame_at(t)).copy()
        c = cam.at(t)
        out = img.copy()
        K = c.Km
        hw, hh = img.shape[1] / 2 / K[0, 0], img.shape[0] / 2 / K[1, 1]
        for ln in F.field_lines():  # only points inside the field of view (distortion wraps the rest back in)
            dense = np.concatenate([a + (b - a) * s for a, b in zip(ln[:-1], ln[1:]) for s in [np.linspace(0, 1, 60)[:, None]]])
            pc = (c.R @ (dense - c.center_world()).T).T
            ok = (pc[:, 2] > 0.1) & (np.abs(pc[:, 0] / np.maximum(pc[:, 2], 1e-6)) < 1.05 * hw) & \
                 (np.abs(pc[:, 1] / np.maximum(pc[:, 2], 1e-6)) < 1.05 * hh)
            p = c.project(dense)
            for i in range(len(p) - 1):
                if ok[i] and ok[i + 1]:
                    cv2.line(out, tuple(np.int32(p[i])), tuple(np.int32(p[i + 1])), (0, 255, 255), 2)
        for al, col in (("blue", (255, 120, 0)), ("red", (0, 0, 255))):
            cv2.polylines(out, [np.int32(c.project(F.hub_opening_3d(al, 24)))], True, col, 2)
        for i, tg in tags.items():  # tag outlines where the camera says they are (in front of it)
            pts = F.tag_corners(tg)
            if (((c.R @ (pts - c.center_world()).T).T)[:, 2] > 0.1).all():
                p = c.project(pts)
                if np.all(np.abs(p) < 4000):
                    cv2.polylines(out, [np.int32(p)], True, (0, 255, 0), 1)
        out = cv2.resize(out, (480, int(480 * out.shape[0] / out.shape[1])))
        cv2.putText(out, f"{key} {vname} t={t:.0f}s ({getattr(cam, 'recovered', 'tags')})", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(out)
    h = min(x.shape[0] for x in tiles)
    rows.append(np.hstack([x[:h] for x in tiles]))
    vid.close()
    del md
w = max(r.shape[1] for r in rows)
sheet = np.vstack([np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows])
(ROOT / "data" / "exp_calib" / "img").mkdir(exist_ok=True)
cv2.imwrite(str(ROOT / "data" / "exp_calib" / "img" / "recovered_views.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("wrote", sheet.shape)
