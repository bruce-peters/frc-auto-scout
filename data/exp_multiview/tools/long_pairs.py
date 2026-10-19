"""Diagnostic: best joint-fit rms of end-view launch tracks paired with a main-view track over >= 6 common
instants (long, clean overlaps), with the stored main camera and with the landmark camera (lmcam.py), and the same
search against main tracks shifted by 7 s (chance level)."""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmcam  # noqa: E402
import mv_pairs  # noqa: E402
from autoscout import config, triangulate as T  # noqa: E402

k = sys.argv[1] if len(sys.argv) > 1 else "2026cc_qm24"
with open(config.OUTPUT / k / "scan.pkl", "rb") as fh:
    md = pickle.loads(fh.read())
orig_cam = md.cams["main"]
orig_pts = [list(tr.pts) for tr in md.tracks_all["main"]]
u = np.unique(np.concatenate([np.array(p)[:, 0] for p in orig_pts]))
step = float(np.median(np.diff(u)))
T.PAIR_GAP_M, T.PAIR_MAX_CANDS = 1.5, 30
for cam_name in ("stored", "landmark"):
    md.cams["main"] = orig_cam if cam_name == "stored" else lmcam.lm_main_track(orig_cam)
    for shift in (0.0, round(7.0 / step) * step):
        for tr, p in zip(md.tracks_all["main"], orig_pts):
            tr.pts = [(q[0] + shift, *q[1:]) for q in p]
        mv = T.MultiView(md)
        anc = mv_pairs.anchors_of(mv)
        best = []
        for va in ("blue_end", "red_end"):
            for i in anc[va]:
                b = None
                for g, ib, tc, mid in mv.candidates(va, i, "main"):
                    if len(tc) < 6:
                        continue
                    c = mv_pairs.fit_candidate(mv, va, i, "main", ib, tc)
                    if c is not None and (b is None or max(c["rms"]) < b[0]):
                        b = (max(c["rms"]), c["rms"])
                if b is not None:
                    best.append(b)
        r = np.array([b[0] for b in best])
        re = np.array([b[1][0] for b in best]); rm = np.array([b[1][1] for b in best])
        print(f"{cam_name:9s} shift {shift:4.1f}s: end anchors with a >=6-instant partner {len(r)}; best max-rms <=4 {np.sum(r <= 4)}, <=6 {np.sum(r <= 6)}, <=10 {np.sum(r <= 10)};"
              f" median end rms {np.median(re):.1f} main rms {np.median(rm):.1f}", flush=True)
