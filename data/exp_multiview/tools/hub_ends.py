"""Diagnostic: where descending ball tracks end (vanish into a hub) relative to the projected hub-opening centre,
in each view, in metres at the hub (x right / y down in the image). A cheap above-the-floor landmark check.

  .venv/Scripts/python data/exp_multiview/tools/hub_ends.py 2026cc_qm24
"""
import pickle, sys, numpy as np
sys.path.insert(0, r"C:\Users\bruce\Documents\Coding\auto-scout")
from autoscout import config, field as F
k = sys.argv[1]
with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
    md = pickle.loads(fh.read())
for v, trs in md.tracks_all.items():
    cam = md.cams[v]
    for al in ("blue", "red"):
        ends = []
        for tr in trs:
            if len(tr.pts) < 5: continue
            p = np.array(tr.pts)
            c = cam.at(p[-1, 0])
            hc = c.project(np.array([[*F.hub_center(al), config.HUB_OPENING_HEIGHT]]))[0]
            hb = c.project(np.array([[*F.hub_center(al), 0.0]]))[0]
            scale = np.linalg.norm(hc - hb) / config.HUB_OPENING_HEIGHT  # px per m vertical
            d = p[-1, 1:3] - hc
            if np.linalg.norm(d) > 1.5 * scale: continue
            if p[-1, 2] - p[-3, 2] <= 0: continue  # descending at the end
            ends.append((d[0] / scale, d[1] / scale))
        if len(ends) < 20: continue
        e = np.array(ends)
        H, xe, ye = np.histogram2d(e[:, 0], e[:, 1], bins=12, range=[[-1.5, 1.5], [-1.5, 1.5]])
        i, j = np.unravel_index(np.argmax(H), H.shape)
        print(f"{v:9s} {al:4s} n={len(e):4d} median end offset (m at hub, x right / y down) {np.median(e[:,0]):+.2f} {np.median(e[:,1]):+.2f}  mode cell {0.5*(xe[i]+xe[i+1]):+.2f} {0.5*(ye[j]+ye[j+1]):+.2f}  px/m {scale:.0f}")
