"""cc_qm24 70-74 s: 254's boxes per view, blobs near it, and what each tracker makes of them."""
import json, pickle, sys
from collections import Counter
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from autoscout import fuel, pipeline, views
import stream_experiment as S
key, team, t_lo, t_hi = "2026cc_qm24", 254, 70.0, 74.0
dets = json.load(open(ROOT / "output" / key / "dets.json"))
md = pickle.load(open(ROOT / "output" / key / "scan.pkl", "rb"))
vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
log = pickle.load(open(ROOT / "data/exp_fueldet/blobs" / f"{key}.pkl", "rb"))
def reason(tr, ppm):
    if len(tr.pts) < 4: return "short"
    vu, vv = tr.velocity0()
    if np.hypot(vu, vv) / max(ppm, 1e-3) < 3.5: return "slow"
    if vv > -0.25 * abs(vu) and vv > -1.0 * ppm: return "not rising"
    if tr.fit()["res"] >= 6.0: return "bad fit"
    return "shot"
for vname in vl:
    ox, oy = vl[vname].offset(1920, 1080)
    B = np.array([(d[0], d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy) for d in dets if d[7] == team and d[1] == vname and t_lo <= d[0] <= t_hi])
    if not len(B):
        print(vname, "no 254 box"); continue
    x0, y0, x1, y1 = np.median(B[:, 1:], axis=0); bw, bh = x1 - x0, y1 - y0
    near = lambda u, v: x0 - 1.5 * bw <= u <= x1 + 1.5 * bw and y0 - 6 * bh <= v <= y1
    rows = [r for r in log["blob_log"][vname] if t_lo <= r[0] <= t_hi]
    nb = np.mean([sum(near(b[0], b[1]) for b in r[1]) for r in rows])
    area = np.median([b[2] for r in rows for b in r[1] if near(b[0], b[1])] or [0])
    print(f"{vname}: 254 box {np.round([x0, y0, x1, y1])} ({len(B)} dets), motion blobs near/frame {nb:.1f}, median area {area:.0f}")
    cam = md.cams.get(vname)
    for name in ("motion", "motion+bt", "motion+split"):
        tracks = S.track_log({vname: log["blob_log"][vname]}, 2, **S.VARIANTS[name])[vname]
        T = [tr for tr in tracks if t_lo <= tr.pts[0][0] <= t_hi and near(tr.pts[0][1], tr.pts[0][2])]
        c = Counter(reason(tr, pipeline._px_per_m(cam.at(tr.pts[0][0]) if cam else None, tr.pts[0][1], tr.pts[0][2])) for tr in T)
        L = [len(tr.pts) for tr in T]
        print(f"   {name:14s} tracks {len(T):4d}  {dict(c)}  pts median {np.median(L) if L else 0:.0f} max {max(L) if L else 0}")
