"""Exit-band yellow flux per robot from the broadcast video (untracked, so dense streams still count).

For every processed frame (30 Hz, like the scan's fuel step: every 2nd frame of 59.94 fps, neighbours +-2 source
frames), the scan's moving-yellow test (fuel.yellow_mask & |grey - prev| > 22 & |grey - next| > 22, open 2x2,
dilate 3x3) inside an exit band above each robot box of the nearest detection frame (<= 0.1 s, v5 dets.json
boxes and teams):
    band x: [x0 - 0.25 w, x1 + 0.25 w], band y: [y0 - 0.9 h, y0 + 0.1 h]   (the analyse step (1) link region)
clipped to the box's view. A pixel inside several bands goes to the box whose centre x is nearest ('flux'); 'fluxall'
counts it for every band. Weight per frame = pixels / one-ball area at that range (pi (0.075 ppm)^2, ppm from the
scan's camera via pipeline._px_per_m), i.e. ball-equivalents x frames in the band.

Reads data/videos/<key>.mp4, output/<key>/scan.pkl (cams, prod; read-only), v5_snapshot dets.json.
Writes data/exp_launch/sig_flux/<key>.pkl  {team: {"flux": [(t, w)], "fluxall": [(t, w)]}}.

  flux.py KEY [KEY ...]      (skips keys already done)
"""
import gc
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
from autoscout import config, fuel, pipeline as P, views  # noqa: E402

SNAP, VID, OUT = ROOT / "data/exp_attrmodel/v5_snapshot", ROOT / "data/videos", ROOT / "data/exp_launch/sig_flux"
OUT.mkdir(parents=True, exist_ok=True)
K_OPEN, K_DIL = np.ones((2, 2), np.uint8), np.ones((3, 3), np.uint8)


def run(k):
    rows = json.loads((SNAP / k / "dets.json").read_text())
    r = json.loads((SNAP / k / "result.json").read_text())
    with open(ROOT / "output" / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cams, prod = md.cams, md.prod
    del md; gc.collect()
    W, H = 1920, 1080
    vl = {v.name: v for v in views.PRODUCTIONS[prod]}
    by_vt = defaultdict(lambda: defaultdict(list))
    for d in rows:
        if d[7] is not None:
            by_vt[d[1]][d[0]].append((d[2:6], d[7]))
    tk = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    tline = r["timeline"]
    t0 = (tline["auto_start"] or P.FIRST_AUTO_START) - 1.0
    t1 = (tline["teleop_start"] or t0 + 24) + config.TELEOP_LEN + 5.0
    cap = cv2.VideoCapture(str(VID / f"{k}.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    f0 = max(0, int(t0 * fps) - 4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    buf = []  # (t, frame, grey) of processed frames
    out = defaultdict(lambda: {"flux": [], "fluxall": []})
    idx = f0
    tic = time.time()
    while True:
        if (idx - f0) % 2:
            ok = cap.grab()
            idx += 1
            if not ok:
                break
            continue
        ok, fr = cap.read()
        idx += 1
        if not ok:
            break
        t = (idx - 1) / fps
        buf.append((t, fr, cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)))
        if len(buf) < 3:
            continue
        if len(buf) > 3:
            buf.pop(0)
        tc, fc, gc_ = buf[1]
        gp, gn = buf[0][2], buf[2][2]
        if tc > t1:
            break
        for vname, ts in tk.items():
            if vname not in vl:
                continue
            j = int(np.argmin(np.abs(ts - tc)))
            if abs(ts[j] - tc) > 0.1:
                continue
            vx0, vy0, vx1, vy1 = vl[vname].scaled(W, H)
            boxes = by_vt[vname][ts[j]]
            bands = []
            for (x0, y0, x1, y1), team in boxes:
                w, h = x1 - x0, y1 - y0
                bx0, bx1 = int(max(vx0, x0 - 0.25 * w)), int(min(vx1, x1 + 0.25 * w))
                by0, by1 = int(max(vy0, y0 - 0.9 * h)), int(min(vy1, y0 + 0.1 * h))
                if bx1 - bx0 >= 2 and by1 - by0 >= 2:
                    bands.append((bx0, by0, bx1, by1, (x0 + x1) / 2, team))
            if not bands:
                continue
            ux0, uy0 = min(b[0] for b in bands), min(b[1] for b in bands)
            ux1, uy1 = max(b[2] for b in bands), max(b[3] for b in bands)
            sl = (slice(uy0, uy1), slice(ux0, ux1))
            mov = (cv2.absdiff(gc_[sl], gp[sl]) > 22) & (cv2.absdiff(gc_[sl], gn[sl]) > 22)
            m = (mov & (fuel.yellow_mask(fc[sl]) > 0)).astype(np.uint8)
            if not m.any():
                continue
            m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, K_OPEN), K_DIL)
            yy, xx = np.nonzero(m)
            if not len(xx):
                continue
            xx, yy = xx + ux0, yy + uy0
            inb = np.array([(xx >= b[0]) & (xx < b[2]) & (yy >= b[1]) & (yy < b[3]) for b in bands])
            dx = np.array([np.abs(xx - b[4]) for b in bands])
            dx[~inb] = np.inf
            own = np.argmin(dx, 0)
            anyb = inb.any(0)
            cam = cams.get(vname)
            ca = cam.at(tc) if cam is not None else None
            ox, oy = vl[vname].offset(W, H)
            for bi, (bx0, by0, bx1, by1, cx, team) in enumerate(bands):
                nall = int(inb[bi].sum())
                if not nall:
                    continue
                nown = int((anyb & (own == bi)).sum())
                ppm = P._px_per_m(ca, cx - ox, by1 - oy)
                a1 = max(np.pi * (0.075 * ppm) ** 2, 1.0)
                out[team]["fluxall"].append((tc, nall / a1))
                if nown:
                    out[team]["flux"].append((tc, nown / a1))
    cap.release()
    with open(OUT / f"{k}.pkl", "wb") as fh:
        pickle.dump(dict(out), fh)
    n = sum(len(v["flux"]) for v in out.values())
    print(f"{k}: {n} flux events, {time.time() - tic:.0f} s", flush=True)


if __name__ == "__main__":
    for k in sys.argv[1:]:
        if (OUT / f"{k}.pkl").exists():
            continue
        if not (VID / f"{k}.mp4").exists():
            print(k, "no video", flush=True)
            continue
        run(k)
