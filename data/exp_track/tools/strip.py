"""Frame strip around a robot: motion blobs (thin green), tracks of one tracker up to that frame (colour per
track, thick = passes is_shot), shooter box (cyan)."""
import json, pickle, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import track_lab as T
from autoscout import config, fuel, pipeline as P, views
from autoscout.video import Video, video_path
key, team, vn, t0, t1, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
variant = sys.argv[8] if len(sys.argv) > 8 else "ballistic+split"
md = pickle.load(open(config.OUTPUT / key / "scan.pkl", "rb")); log = pickle.load(open(T.BLOBS / f"{key}.pkl", "rb"))
dets = json.load(open(config.OUTPUT / key / "dets.json")); vw = next(v for v in views.PRODUCTIONS[md.prod] if v.name == vn)
ox, oy = vw.offset(1920, 1080)
B = np.array([(d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy) for d in dets if d[7] == team and d[1] == vn and t0 - 1 <= d[0] <= t1 + 1])
x0, y0, x1, y1 = np.median(B, axis=0); bw, bh = x1 - x0, y1 - y0
cx0, cx1, cy0, cy1 = int(max(0, x0 - 2.5 * bw)), int(x1 + 2.5 * bw), int(max(0, y0 - 5 * bh)), int(y1 + 0.5 * bh)
tracks = T.run(variant, {vn: log["blob_log"][vn]}, md, log["fuel_step"])[vn]
cam_t = md.cams.get(vn)
shot = {id(tr): fuel.is_shot(tr, P._px_per_m(cam_t.at(tr.pts[0][0]), tr.pts[0][1], tr.pts[0][2])) for tr in tracks}
rng = np.random.default_rng(3); col = {id(tr): tuple(int(c) for c in rng.integers(40, 255, 3)) for tr in tracks}
rows = {round(r[0], 4): r for r in log["blob_log"][vn]}
vid = Video(video_path(key)); tiles = []
for t in np.linspace(t0, t1, n):
    tc = min(rows, key=lambda x: abs(x - t)); _, mb, _ = rows[tc]
    img = vw.crop(vid.frame(int(round(tc * vid.fps)))).copy()
    for b in mb:
        cv2.rectangle(img, (int(b[5]), int(b[6])), (int(b[5] + b[3]), int(b[6] + b[4])), (0, 255, 0), 1)
    for tr in tracks:
        p = np.array([(q[1], q[2]) for q in tr.pts if q[0] <= tc + 1e-6 and q[0] >= tc - 0.5])
        if len(p) >= 2 and tr.pts[-1][0] >= tc - 0.05:
            cv2.polylines(img, [p.astype(np.int32).reshape(-1, 1, 2)], False, col[id(tr)], 3 if shot[id(tr)] else 1)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 0), 1)
    c = img[cy0:cy1, cx0:cx1]; s = 300 / c.shape[0]; c = cv2.resize(c, (int(c.shape[1] * s), 300), interpolation=cv2.INTER_NEAREST)
    cv2.putText(c, f"{tc:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    tiles.append(c)
rows_img = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
w = max(r.shape[1] for r in rows_img); rows_img = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows_img]
cv2.imwrite(out, np.vstack(rows_img))
