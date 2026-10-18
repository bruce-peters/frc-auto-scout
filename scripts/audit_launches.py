"""(Visual audit tool.) Render a sample of linked launches for a visual audit: the frame just after launch, cropped
around the launch point, with the launch arc, every robot box and its assigned team drawn.
Usage: shot_audit.py OUTDIR SEED match1 match2 ...  (writes OUTDIR/audit_XX.jpg + OUTDIR/shots.json)"""
import json, random, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
from autoscout import config  # noqa: E402
from autoscout.inspector import server  # noqa: E402

out, seed, keys = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(seed)
PER = 6
panels, meta = [], []
for key in keys:
    D = server.match_detail(key)
    vbox = {v["name"]: v["box"] for v in D["views"]}
    tl = D["result"]["timeline"]
    a0, ts = tl.get("auto_start") or 6.13, tl.get("teleop_start")
    to_mt = lambda vt: (20 + vt - ts) if ts and vt >= ts else vt - a0
    # credited launches only: own-hub direction, inside the match
    shots = [s for s in D["result"]["shots"] if s["team"] is not None and s["toward"] and 0 <= to_mt(s["t"]) <= 160]
    pick = rng.sample(shots, min(PER, len(shots)))
    dets = D["dets"]
    det_t = np.array([d[0] for d in dets])
    cap = cv2.VideoCapture(str(config.VIDEOS / f"{key}.mp4"))
    W, H = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    sx, sy = W / 1920, H / 1080
    for s in pick:
        k = min(5, len(s["pts"]) - 1)
        t_show = s["pts"][k][0]
        cap.set(cv2.CAP_PROP_POS_MSEC, t_show * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        ox, oy = vbox[s["view"]][0] * sx, vbox[s["view"]][1] * sy
        # boxes at the nearest detection time
        j = int(np.argmin(np.abs(det_t - s["t"])))
        tj = det_t[j]
        near = [d for d in dets if abs(d[0] - tj) < 1e-6]
        for d in near:
            x0, y0, x1, y1 = [int(round(c)) for c in (d[2] * sx, d[3] * sy, d[4] * sx, d[5] * sy)]
            col = (60, 70, 240) if d[6] == "red" else (230, 160, 90)
            cv2.rectangle(fr, (x0, y0), (x1, y1), col, 2)
            cv2.putText(fr, str(d[7]) if d[7] is not None else "?", (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(fr, str(d[7]) if d[7] is not None else "?", (x0, max(12, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        pts = np.array([[p[1] + ox, p[2] + oy] for p in s["pts"]], np.int32)
        cv2.polylines(fr, [pts], False, (0, 255, 255), 2)
        lu, lv = pts[0]
        cv2.circle(fr, (int(lu), int(lv)), 7, (0, 255, 0), 2)
        # crop around the launch point (and the arc start)
        cw, ch = 520, 340
        cx0 = int(np.clip(lu - cw / 2, 0, W - cw)); cy0 = int(np.clip(lv - ch * 0.45, 0, H - ch))
        crop = cv2.resize(fr[cy0:cy0 + ch, cx0:cx0 + cw], (780, 510))
        n = len(meta)
        cap_txt = f"#{n} {key} mt {to_mt(s['t']):.1f}s  linked {s['team']} ({s['alliance']}) via {s['how']} conf {s['conf']:.2f} {s['view']}"
        cv2.rectangle(crop, (0, 0), (780, 26), (0, 0, 0), -1)
        cv2.putText(crop, cap_txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        panels.append(crop)
        meta.append({"n": n, "key": key, "t": s["t"], "mt": round(to_mt(s["t"]), 2), "team": s["team"], "how": s["how"],
                     "conf": s["conf"], "view": s["view"], "u": s["u"], "v": s["v"]})
    cap.release()
for i in range(0, len(panels), 4):
    grp = panels[i:i + 4] + [np.zeros_like(panels[0])] * (4 - len(panels[i:i + 4]))
    grid = np.vstack([np.hstack(grp[:2]), np.hstack(grp[2:])])
    cv2.imwrite(str(out / f"audit_{i // 4:02d}.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 88])
(out / "shots.json").write_text(json.dumps(meta, indent=1))
print(len(meta), "shots rendered to", out)
