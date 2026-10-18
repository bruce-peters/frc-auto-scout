"""(Visual audit tool.) Render a random sample of launches the shot gate REJECTS, to estimate how many real shots it drops."""
import json, pickle, random, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
from autoscout import config, pipeline, views  # noqa: E402

out, seed, keys = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(seed)
pipeline.SHOT_GATE = True
REJ = []
ob, op = pipeline._box_shot_ok, pipeline._proj_shot_ok
pipeline._box_shot_ok = lambda tr, box: ob(tr, box) or REJ.append(("box", tr, box)) or False
pipeline._proj_shot_ok = lambda tr, ppm: op(tr, ppm) or REJ.append(("proj", tr, None)) or False
panels, meta = [], []
for key in keys:
    REJ.clear()
    md = pickle.load(open(config.OUTPUT / key / "scan.pkl", "rb"))
    pipeline.analyse(md, pipeline.Options())
    tl = md.timeline if hasattr(md, "timeline") else None
    rej = [r for r in REJ if r[1].pts[0][0] > 5]
    pick = rng.sample(rej, min(4, len(rej)))
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    cap = cv2.VideoCapture(str(config.VIDEOS / f"{key}.mp4"))
    W, H = cap.get(3), cap.get(4)
    for how, tr, box in pick:
        vname = next((n for n, trs in md.tracks_all.items() if any(t is tr for t in trs)), "main")
        ox, oy = vl[vname].offset(1920, 1080)
        k = min(5, len(tr.pts) - 1)
        cap.set(cv2.CAP_PROP_POS_MSEC, tr.pts[k][0] * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        pts = np.array([[p[1] + ox, p[2] + oy] for p in tr.pts], np.int32)
        if box is not None:
            x0, y0, x1, y1 = box
            cv2.rectangle(fr, (int(x0 + ox), int(y0 + oy)), (int(x1 + ox), int(y1 + oy)), (255, 255, 255), 2)
        cv2.polylines(fr, [pts], False, (0, 255, 255), 2)
        cv2.circle(fr, tuple(int(c) for c in pts[0]), 7, (0, 0, 255), 2)
        cw, ch = 520, 340
        lu, lv = pts[0]
        cx0 = int(np.clip(lu - cw / 2, 0, W - cw)); cy0 = int(np.clip(lv - ch * 0.45, 0, H - ch))
        crop = cv2.resize(fr[cy0:cy0 + ch, cx0:cx0 + cw], (780, 510))
        n = len(meta)
        cv2.rectangle(crop, (0, 0), (780, 26), (0, 0, 0), -1)
        cv2.putText(crop, f"#{n} {key} t {tr.pts[0][0]:.1f}s REJECTED ({how}) {vname} npts {len(tr.pts)}", (6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(crop)
        meta.append({"n": n, "key": key, "t": tr.pts[0][0], "how": how, "view": vname})
    cap.release()
for i in range(0, len(panels), 4):
    grp = panels[i:i + 4] + [np.zeros_like(panels[0])] * (4 - len(panels[i:i + 4]))
    cv2.imwrite(str(out / f"rej_{i // 4:02d}.jpg"), np.vstack([np.hstack(grp[:2]), np.hstack(grp[2:])]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
(out / "rejected.json").write_text(json.dumps(meta, indent=1))
print(len(meta), "rejected launches rendered")
