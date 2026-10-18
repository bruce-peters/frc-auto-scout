"""Where do a burst's balls get lost? For every track starting in the burst window near the shooter (view-local
region above its box): shot test reason, exclusive owner, toward-hub. One process, one match."""
import json, pickle, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import track_lab as T
from autoscout import config, fuel, field as fieldmod, pipeline as P, views
key, team, b0, b1 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
variants = sys.argv[5:] or ["ballistic+split"]
md = pickle.load(open(config.OUTPUT / key / "scan.pkl", "rb"))
log = pickle.load(open(T.BLOBS / f"{key}.pkl", "rb"))
dets = json.load(open(config.OUTPUT / key / "dets.json"))
vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
al = next(d[6] for d in dets if d[7] == team)
def reason(tr, ppm):
    if len(tr.pts) < 4: return "short"
    vu, vv = tr.velocity0()
    if np.hypot(vu, vv) / max(ppm, 1e-3) < 3.5: return "slow"
    if vv > -0.25 * abs(vu) and vv > -1.0 * ppm: return "not rising"
    if tr.fit()["res"] >= 6.0: return "bad fit"
    return "shot"
for name in variants:
    tracks = T.run(name, log["blob_log"], md, log["fuel_step"])
    for vn in ("main", f"{al}_end"):
        ox, oy = vl[vn].offset(1920, 1080)
        B = np.array([(d[0], d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy) for d in dets if d[7] == team and d[1] == vn and b0 - 1 <= d[0] <= b1 + 1])
        if not len(B):
            print(name, vn, "shooter not visible"); continue
        x0, y0, x1, y1 = np.median(B[:, 1:], axis=0); bw, bh = x1 - x0, y1 - y0
        cam_t = md.cams.get(vn)
        others = {}
        for d in dets:
            if d[1] == vn and d[7] is not None and abs(d[0] - (b0 + b1) / 2) < 2:
                others.setdefault(d[7], []).append((d[0], d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy))
        c = Counter()
        for tr in tracks[vn]:
            t0 = tr.pts[0][0]
            if not (b0 - 1.5 <= t0 <= b1 + 1.5): continue
            u0, v0 = tr.pts[0][1], tr.pts[0][2]
            if not (x0 - 1.5 * bw <= u0 <= x1 + 1.5 * bw and y0 - 5 * bh <= v0 <= y1): continue
            cam = cam_t.at(t0) if cam_t else None
            ppm = P._px_per_m(cam, u0, v0)
            rs = reason(tr, ppm)
            if rs != "shot":
                c[rs] += 1; continue
            # owner among all robots with a box near t0
            best = None
            for tm, bx in others.items():
                bx = np.array(sorted(bx)); j = int(np.argmin(np.abs(bx[:, 0] - t0)))
                if abs(bx[j, 0] - t0) > 0.3: continue
                box = bx[j, 1:]; lk = T.launch_link(tr, box)
                if lk is None: continue
                cost = abs(lk[1] - 0.5 * (box[0] + box[2])) / max(box[2] - box[0], 1)
                if best is None or cost < best[0]: best = (cost, tm)
            if best is None: c["shot, traces to no box"] += 1; continue
            if best[1] != team: c[f"shot, owner {best[1]}"] += 1; continue
            hu, hv = cam.project(np.array([[*fieldmod.hub_center(al), 1.9]]))[0]
            tw = np.hypot(tr.pts[-1][1] - hu, tr.pts[-1][2] - hv) < np.hypot(u0 - hu, v0 - hv) - 0.3 * ppm
            c["counted" if tw else "shot, not toward hub"] += 1
        print(f"{name:22s} {vn:9s} box {np.round([x0, y0, x1, y1])} ppm~{ppm:.0f} | {dict(c.most_common())}")
