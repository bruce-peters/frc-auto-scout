"""Show what the in-flight fuel detector sees: run the pipeline's motion-blob detection (fuel.view_blobs, three-frame
differencing on the FUEL-yellow mask, same frame offsets as FuelScan) on one frame and circle every detected blob.
Yellow balls in the air without a circle are misses.

  .venv/Scripts/python data/exp_audit/tools/blob_frames.py MATCH T VIEW --out FILE [--crop x0,y0,x1,y1] [--zoom 2]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import fuel, pipeline, views  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("match")
ap.add_argument("t", type=float)
ap.add_argument("view")
ap.add_argument("--out", required=True)
ap.add_argument("--crop", default=None, help="view-local x0,y0,x1,y1")
ap.add_argument("--zoom", type=float, default=2.0)
a = ap.parse_args()

v = Video(video_path(a.match))
W, H = v.width, v.height
fps = v.fps if getattr(v, "fps", None) else 60.0
k = pipeline.Options().fuel_k / fps  # FuelScan's neighbour offset in seconds
f_prev, f_cur, f_next = v.frame_at(a.t - k), v.frame_at(a.t), v.frame_at(a.t + k)
prod = views.detect_production(f_cur)
vw = next(x for x in views.PRODUCTIONS[prod] if x.name == a.view)
x0, y0, x1, y1 = vw.scaled(W, H)
valid = np.ones((y1 - y0, x1 - x0), np.uint8)
for mx0, my0, mx1, my1 in vw.mask_boxes(W, H):
    valid[max(0, my0):max(0, my1), max(0, mx0):max(0, mx1)] = 0
g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1] for f in (f_prev, f_cur, f_next)]
ymask = fuel.yellow_mask(f_cur)[y0:y1, x0:x1]
blobs = fuel.view_blobs(g[1], g[0], g[2], ymask, valid)
img = f_cur[y0:y1, x0:x1].copy()
for b in blobs:
    r = max(6, int(0.6 * max(b.w, b.h)) + 4)
    cv2.circle(img, (int(b.u), int(b.v)), r, (0, 255, 0), 2)
cx0, cy0, cx1, cy1 = [int(c) for c in a.crop.split(",")] if a.crop else (0, 0, img.shape[1], img.shape[0])
img = img[cy0:cy1, cx0:cx1]
img = cv2.resize(img, None, fx=a.zoom, fy=a.zoom, interpolation=cv2.INTER_CUBIC)
cv2.putText(img, f"{a.match} {a.view} t={a.t:.2f}s: {len(blobs)} moving-yellow blobs detected (green)", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(a.out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(a.out, len(blobs), "blobs")
