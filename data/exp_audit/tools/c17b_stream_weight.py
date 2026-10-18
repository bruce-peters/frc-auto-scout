"""Cycle 17b: give a credited launch the weight of the balls it stands for.

c17: in volleys, launch arcs per scouted ball fall 4x (merged blobs: median track blob area 32 -> 38 px,
5026 29 -> 47; more 'not smooth' rejections). Two per-launch multipliers, from the scan's flight tracks:
  area     : m_a = clip(area_ratio / median(area_ratio), 1, AMAX); area_ratio = median blob area of the
             track / pi*(0.075*ppm)^2 (a single 15 cm ball at that range)
  companion: m_c = 1 + C * (other fast rising tracks starting within 0.3 s and 0.6 m (at the launch
             point's px/m) of this launch that were not themselves credited)
shot_w *= m. Re-split from result.json; scored vs the cc scouting and fuel OPR.
"""
import gc
import json
import pickle

import numpy as np

from common import AUDIT, DIRS, P, load, match_keys, resplit, scout_metrics, share_metrics, tba, team_signals
from autoscout import fuel

R, INFO, FEAT = {}, {}, {}
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k)
    R[k], INFO[k] = r, info
    with open(DIRS["v2"] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    idx = {}
    starts = {}
    for vname, tracks in md.tracks_all.items():
        cam_track = md.cams.get(vname)
        st = []
        for tr in tracks:
            if len(tr.pts) < 4:
                continue
            t0, u0, v0 = tr.pts[0][0], tr.pts[0][1], tr.pts[0][2]
            idx[(vname, round(t0, 3), round(u0, 1), round(v0, 1))] = tr
            vu, vv = tr.velocity0()
            cam = cam_track.at(t0) if cam_track is not None else None
            ppm = P._px_per_m(cam, u0, v0)
            if vv < 0 and np.hypot(vu, vv) / max(ppm, 1e-3) >= 3.5:
                st.append((t0, u0, v0))
        starts[vname] = np.array(st).reshape(-1, 3)
    cams = md.cams
    del md
    gc.collect()
    credited = {(s["view"], round(s["t"], 3), round(s["u"], 1), round(s["v"], 1)) for s in r["shots"] if s["team"] is not None}
    feats = []
    for s in r["shots"]:
        key = (s["view"], round(s["t"], 3), round(s["u"], 1), round(s["v"], 1))
        tr = idx.get(key)
        cam_track = cams.get(s["view"])
        cam = cam_track.at(s["t"]) if cam_track is not None else None
        ppm = P._px_per_m(cam, s["u"], s["v"])
        if tr is None:
            feats.append((np.nan, 0)); continue
        area = float(np.median([q[3] for q in tr.pts]))
        ratio = area / (np.pi * (0.075 * ppm) ** 2)
        S = starts.get(s["view"])
        comp = 0
        if S is not None and len(S):
            m = (np.abs(S[:, 0] - s["t"]) <= 0.3) & (np.hypot(S[:, 1] - s["u"], S[:, 2] - s["v"]) <= 0.6 * ppm)
            for t0, u0, v0 in S[m]:
                if (s["view"], round(t0, 3), round(u0, 1), round(v0, 1)) not in credited:
                    comp += 1
        feats.append((ratio, comp))
    FEAT[k] = feats
    print(k, "done", flush=True)
allr = np.array([f[0] for k in FEAT for f, s in zip(FEAT[k], R[k]["shots"]) if s["team"] is not None and np.isfinite(f[0])])
med = float(np.median(allr))
comp = np.array([f[1] for k in FEAT for f, s in zip(FEAT[k], R[k]["shots"]) if s["team"] is not None])
print(f"credited launches: area ratio median {med:.2f} q90 {np.quantile(allr, .9):.2f}; companions mean {comp.mean():.2f}, "
      f"share with >= 1 {np.mean(comp >= 1):.2f}")


def run(amax=1.0, C=0.0):
    pm = {}
    for k in R:
        r = json.loads(json.dumps(R[k]))
        for s, (ratio, cp) in zip(r["shots"], FEAT[k]):
            m = 1.0
            if amax > 1 and np.isfinite(ratio):
                m *= float(np.clip(ratio / med, 1.0, amax))
            m *= 1.0 + C * cp
            s["conf"] = s["conf"] * m
        pm[k] = resplit(r, INFO[k])
    return pm


res = {}
for name, kw in (("base", {}), ("area<=2", {"amax": 2.0}), ("area<=4", {"amax": 4.0}), ("comp0.5", {"C": 0.5}),
                 ("comp1", {"C": 1.0}), ("area<=3+comp0.5", {"amax": 3.0, "C": 0.5})):
    pm = run(**kw)
    res[name] = pm
    print(f"{name:16s}", share_metrics(pm), scout_metrics(pm), flush=True)
# 5026 at cc under each variant
from common import scout_counts
sc = scout_counts()
for name, pm in res.items():
    gaps = []
    for k, f in pm.items():
        if k in sc and 5026 in sc[k]:
            info = INFO[k]; al = "red" if 5026 in info.red else "blue"; tm = info.teams(al)
            if all(t in sc[k] for t in tm):
                v = np.array([f[t] for t in tm]); s = np.array([sc[k][t] for t in tm])
                gaps.append(v[tm.index(5026)] / v.sum() - s[tm.index(5026)] / s.sum())
    print(f"   5026 share gap vs scouting ({name}): {np.mean(gaps):+.3f} (n={len(gaps)})")
json.dump({"fuel": {n: {k: {str(t): f for t, f in v.items()} for k, v in pm.items()} for n, pm in res.items()}},
          open(AUDIT / "c17b_stream_weight.json", "w"))
