"""Hypothesis test: are true cross-view pairs hidden by a main-camera 3-D calibration error?

The Chezy main camera sees four coplanar tags 20-30 m away, so its distance / focal length / tilt are weakly
determined (data/exp_maincal). If that is why main + end pairs fit at 8-20 px instead of ~3 px, a small rigid
correction of the main camera (rotation, centre, focal length; the end cameras fixed) should make many more
anchors find a <= 4 px partner. Candidate pairs are collected once with a generous ray gap and re-fitted for each
perturbed main camera; the objective is the sum over anchors of the truncated squared rms of the best partner.
A time-shifted copy of the main tracks (7 s) gives the chance level for the same search.

  .venv/Scripts/python data/exp_multiview/tools/selfcal_test.py 2026cc_qm24 [--shift]
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import mv_pairs  # noqa: E402
from autoscout import config, triangulate as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("key")
ap.add_argument("--shift", action="store_true")
ap.add_argument("--maxiter", type=int, default=400)
ap.add_argument("--lm", action="store_true", help="start from the landmark main camera (lmcam.py)")
a = ap.parse_args()
with open(config.OUTPUT / a.key / "scan.pkl", "rb") as fh:
    md = pickle.loads(fh.read())
if a.lm:
    import lmcam  # noqa: E402
    md.cams["main"] = lmcam.lm_main_track(md.cams["main"])
    print("landmark main camera, pose rms", round(md.cams["main"].rms, 2), "px", flush=True)
if a.shift:
    u = np.unique(np.concatenate([np.array(tr.pts)[:, 0] for tr in md.tracks_all["main"]]))
    step = float(np.median(np.diff(u)))
    sh = round(7.0 / step) * step
    for tr in md.tracks_all["main"]:
        tr.pts = [(p[0] + sh, *p[1:]) for p in tr.pts]
T.PAIR_GAP_M = 1.5
T.PAIR_MAX_CANDS = 20
mv = T.MultiView(md)
anc = mv_pairs.anchors_of(mv)
pairs = []  # (anchor id, end obs (sub), main obs (sub), main uv)
for va in ("blue_end", "red_end"):
    for i in anc.get(va, []):
        oa = mv.v[va].obs(i)
        for g, ib, tc, mid in mv.candidates(va, i, "main"):
            ob = mv.v["main"].obs(ib)
            lo, hi = float(tc.min()) - 0.1 - 1e-3, float(tc.max()) + 0.1 + 1e-3
            sa = T._sub(oa, oa["t"] <= hi)
            sb = T._sub(ob, (ob["t"] >= lo) & (ob["t"] <= hi))
            pairs.append(((va, i), sa, sb))
print(len(pairs), "candidate pairs for", len({p[0] for p in pairs}), "end-view anchors", flush=True)
ct = md.cams["main"]
K0 = np.asarray(ct.K, float)
D0 = np.asarray(ct.dist, float)
f0 = 0.5 * (K0[0, 0] + K0[1, 1])
# per main observation: raw uv and the pose at its time
for _, sa, sb in pairs:
    sb["C0"] = sb["C"].copy()
    sb["R0"] = sb["R"].copy()


def apply(p):
    """p = [rx, ry, rz (deg, camera frame), cx, cy, cz (m, world), df (fraction of f), dt (s, main time shift)]."""
    dR = cv2.Rodrigues(np.radians(p[:3]))[0]
    K = K0.copy()
    K[0, 0] *= 1 + p[6]
    K[1, 1] *= 1 + p[6]
    out = []
    for key, sa, sb in pairs:
        R = np.einsum("ij,njk->nik", dR, sb["R0"])
        C = sb["C0"] + p[3:6]
        tv = -np.einsum("nij,nj->ni", R, C)
        nrm = cv2.undistortPoints(sb["uv"].reshape(-1, 1, 2), K, D0).reshape(-1, 2)
        sb2 = dict(sb)
        sb2.update(R=R, tv=tv, nrm=nrm, f=f0 * (1 + p[6]), t=sb["t"] + p[7])
        out.append((key, sa, sb2))
    return out


def evaluate(p, trunc=8.0):
    best = {}
    for key, sa, sb in apply(p):
        f = T.fit_linear([sa, sb], float(sa["t"][0]), iters=2)
        if f is None:
            continue
        r = max(f["rms"])
        best[key] = min(best.get(key, 1e9), r)
    b = np.array(list(best.values()))
    return float(np.sum(np.minimum(b, trunc) ** 2)), b


tic = time.time()
obj0, b0 = evaluate(np.zeros(8))
print(f"start: objective {obj0:.0f}  anchors with partner <=3px {np.sum(b0 <= 3)} <=4px {np.sum(b0 <= 4)} <=6px {np.sum(b0 <= 6)}"
      f"  ({time.time() - tic:.1f}s per evaluation)", flush=True)
scale = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.02, 0.01])
if a.maxiter <= 0:
    sys.exit(0)
n_eval = [0]


def fun(z):
    n_eval[0] += 1
    return evaluate(z * scale)[0]


res = minimize(fun, np.zeros(8), method="Nelder-Mead", options={"maxiter": a.maxiter, "xatol": 0.05, "fatol": 1.0,
                                                                  "initial_simplex": np.vstack([np.zeros(8), 2 * np.eye(8)])})
p = res.x * scale
obj, b = evaluate(p)
print(f"after {n_eval[0]} evaluations: correction rot {np.round(p[:3], 3)} deg, centre {np.round(p[3:6], 2)} m, f x{1 + p[6]:.3f}, dt {1000 * p[7]:+.1f} ms")
print(f"objective {obj:.0f}  anchors with partner <=3px {np.sum(b <= 3)} <=4px {np.sum(b <= 4)} <=6px {np.sum(b <= 6)}"
      f"  ({time.time() - tic:.0f}s)", flush=True)
