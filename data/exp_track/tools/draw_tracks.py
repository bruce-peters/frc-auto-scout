"""Draw tracks of several trackers over a crop around a robot for a time window (one panel per tracker)."""
import argparse, json, pickle, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "data/exp_track/tools"))
from autoscout import fuel, pipeline, views
from autoscout.video import Video, video_path
ap = argparse.ArgumentParser()
ap.add_argument("key"); ap.add_argument("team", type=int); ap.add_argument("view"); ap.add_argument("t0", type=float); ap.add_argument("t1", type=float)
ap.add_argument("--trackers", nargs="*", default=["motion", "motion+bt", "motion+split"])
ap.add_argument("--out", required=True)
a = ap.parse_args()
import stream_experiment as S
try:
    import track_lab as TL  # new trackers (data/exp_track/tools/track_lab.py), if present
except ImportError:
    TL = None
dets = json.load(open(ROOT / "output" / a.key / "dets.json"))
md = pickle.load(open(ROOT / "output" / a.key / "scan.pkl", "rb"))
v = next(x for x in views.PRODUCTIONS[md.prod] if x.name == a.view); ox, oy = v.offset(1920, 1080)
B = np.array([(d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy) for d in dets if d[7] == a.team and d[1] == a.view and a.t0 <= d[0] <= a.t1])
x0, y0, x1, y1 = np.median(B, axis=0); bw, bh = x1 - x0, y1 - y0
cx0, cx1 = int(max(0, x0 - 4 * bw)), int(x1 + 4 * bw); cy0, cy1 = int(max(0, y0 - 6 * bh)), int(y1 + bh)
log = pickle.load(open(ROOT / "data/exp_fueldet/blobs" / f"{a.key}.pkl", "rb"))
vid = Video(video_path(a.key)); tm = (a.t0 + a.t1) / 2
frame = v.crop(vid.frame_at(tm))
cam = md.cams.get(a.view)
panels = []
rng = np.random.default_rng(1)
for name in a.trackers:
    if name in S.VARIANTS:
        tracks = S.track_log({a.view: log["blob_log"][a.view]}, 2, **S.VARIANTS[name])[a.view]
    else:
        tracks = TL.run(name, {a.view: log["blob_log"][a.view]}, md)[a.view]
    img = frame.copy()
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 1)
    n = 0
    for tr in tracks:
        if not (a.t0 <= tr.pts[0][0] <= a.t1): continue
        p = np.array([(q[1], q[2]) for q in tr.pts])
        if not ((p[:, 0] > cx0) & (p[:, 0] < cx1) & (p[:, 1] > cy0) & (p[:, 1] < cy1)).any(): continue
        ppm = pipeline._px_per_m(cam.at(tr.pts[0][0]) if cam else None, p[0, 0], p[0, 1])
        shot = fuel.is_shot(tr, ppm)
        col = (0, 255, 0) if shot else tuple(int(c) for c in rng.integers(60, 200, 3))
        cv2.polylines(img, [p.astype(np.int32).reshape(-1, 1, 2)], False, col, 2 if shot else 1)
        if shot:
            cv2.circle(img, tuple(p[0].astype(int)), 3, (0, 0, 255), -1); n += 1
    crop = img[cy0:cy1, cx0:cx1]
    s = 420 / crop.shape[0]
    crop = cv2.resize(crop, (int(crop.shape[1] * s), 420))
    cv2.putText(crop, f"{name}: {n} shots (green)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    panels.append(crop)
cv2.imwrite(a.out, np.hstack(panels))
