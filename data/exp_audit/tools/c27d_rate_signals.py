"""Cycle 27d: candidate volley signals scored against the scoreboard (sole counter bursts, all matches).

Per flight track (scan.pkl): box-linked robot (dets.json team), rising, is_shot, credited, points, summed blob
area (as c27). Per sole counter burst (exactly one alliance robot with credited launches in [start - 5.5,
end - 0.8]): that robot's signals in the same window; per counted ball by scoring-rate bin. The flattest
per-ball profile (ratio of the 9-13 to the 0-4 balls/s bin closest to 1, log-log slope vs balls closest to
1) is the best volume signal. Writes data/exp_audit/c27d_bursts.json.
"""
import gc
import json
import pickle
from collections import defaultdict

import numpy as np

from common import A, AUDIT, DIRS, P, TL, load, match_keys, tba
from autoscout import fuel, views

W, H = 1920, 1080
rows = []
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    dets = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    with open(DIRS["v2"] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    cams = md.cams
    by_vt = defaultdict(lambda: defaultdict(list))
    for d in dets:
        by_vt[d[1]][d[0]].append(d)
    tk = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    credited = {(s["view"], round(s["t"], 3), round(s["u"], 1), round(s["v"], 1)) for s in r["shots"] if s["team"] is not None}
    trk = defaultdict(list)
    for vname, tracks in md.tracks_all.items():
        if vname not in tk:
            continue
        ox, oy = vl[vname].offset(W, H)
        cam_t = cams.get(vname)
        ts = tk[vname]
        for tr in tracks:
            if len(tr.pts) < 4:
                continue
            t0, u0, v0 = tr.pts[0][0], tr.pts[0][1], tr.pts[0][2]
            j = int(np.argmin(np.abs(ts - t0)))
            if abs(ts[j] - t0) > 0.2:
                continue
            vu, vv = tr.velocity0()
            lu, lv = u0 - vu * 2 / 60, v0 - vv * 2 / 60
            best, bd, bs = None, 1e9, 1e9
            for d in by_vt[vname][ts[j]]:
                x0, y0, x1, y1 = d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy
                bw, bh = x1 - x0, y1 - y0
                dx = max(x0 - 0.25 * bw - lu, 0, lu - x1 - 0.25 * bw); dy = max(y0 - 0.9 * bh - lv, 0, lv - y1)
                dist = np.hypot(dx, dy) / max(bh, 1)
                sc_ = dist + P.TOP_TIEBREAK * abs(v0 - y0) / max(bh, 1)
                if sc_ < bs:
                    best, bd, bs = d, dist, sc_
            if best is None or bd > 0.6 or best[7] is None:
                continue
            cam = cam_t.at(t0) if cam_t is not None else None
            ppm = P._px_per_m(cam, u0, v0)
            trk[best[7]].append((t0, bool(vv < 0), fuel.is_shot(tr, ppm),
                                 (vname, round(t0, 3), round(u0, 1), round(v0, 1)) in credited, len(tr.pts),
                                 float(sum(q[3] for q in tr.pts)), ppm))
    del md
    gc.collect()
    cred = defaultdict(list)
    for s in r["shots"]:
        if s["team"] is not None and s["toward"]:
            cred[s["team"]].append(s["t"])
    cred = {t: np.sort(v) for t, v in cred.items()}
    trk = {t: (np.array([x[0] for x in v]), v) for t, v in trk.items()}
    for al in ("red", "blue"):
        if not tl.counts.get(al):
            continue
        inc = A.increments_from_counts(P._match_counts(tl, al, info.hub(al) or {}))
        if not inc:
            continue
        t = np.array([x[0] for x in inc]); n = np.array([x[1] for x in inc])
        for b in np.split(np.arange(len(t)), np.where(np.diff(t) > 1.5)[0] + 1):
            bt, bn = t[b], n[b]
            balls = float(bn.sum())
            if balls < 3:
                continue
            a0, a1 = bt[0] - 5.5, bt[-1] - 0.8
            L = {u: ((cred.get(u, np.zeros(0)) >= a0) & (cred.get(u, np.zeros(0)) <= a1)) for u in info.teams(al)}
            act = [u for u, m in L.items() if m.sum() > 0]
            if len(act) != 1:
                continue
            u = act[0]
            c = cred[u][L[u]]
            tt, T = trk.get(u, (np.zeros(0), []))
            sel = [T[i] for i in np.where((tt >= a0) & (tt <= a1))[0]] if len(tt) else []
            dur = sum(cl[-1] - cl[0] + 0.25 for cl in np.split(c, np.where(np.diff(c) > 0.5)[0] + 1)) if len(c) else 0.0
            # ball-equivalents of blob mass: area / single-ball area at that range
            beq = sum(x[5] / max(len(x) and x[4], 1) / max(np.pi * (0.075 * x[6]) ** 2, 1.0) for x in sel if x[1])
            rows.append({"key": k, "balls": balls, "rate": balls / (bt[-1] - bt[0] + 0.33), "credited": int(len(c)),
                         "shot_trk": sum(1 for x in sel if x[2]), "rise_n": sum(1 for x in sel if x[1]),
                         "rise_pts": sum(x[4] for x in sel if x[1]), "rise_mass": sum(x[5] for x in sel if x[1]),
                         "shot_mass": sum(x[5] for x in sel if x[2]), "dur": dur,
                         "span": (c[-1] - c[0] + 0.25) if len(c) else 0.0, "beq": beq})
    print(k, len(rows), flush=True)
json.dump(rows, open(AUDIT / "c27d_bursts.json", "w"), default=float)
SIG = ["credited", "shot_trk", "rise_n", "rise_pts", "rise_mass", "shot_mass", "dur", "span", "beq"]
rate = np.array([x["rate"] for x in rows]); balls = np.array([x["balls"] for x in rows])
BINS = [0, 4, 6, 9, 13]
idx = np.digitize(rate, BINS[1:])  # 0: <4, 1: 4-6, 2: 6-9, 3: 9-13, 4: >13
print(f"\n{len(rows)} sole counter bursts, {balls.sum():.0f} balls")
print(f"{'signal':10s}   <4 b/s   4-6     6-9     9-13    >13    9-13/<4   loglog slope vs balls   spearman vs balls")
rk = lambda a: np.argsort(np.argsort(a))
for s in SIG:
    v = np.array([x[s] for x in rows], float)
    per = [v[idx == i].sum() / max(balls[idx == i].sum(), 1e-9) for i in range(5)]
    pos = v > 0
    sl = np.polyfit(np.log(balls[pos]), np.log(v[pos]), 1)[0]
    sp = np.corrcoef(rk(v), rk(balls))[0, 1]
    print(f"{s:10s} " + " ".join(f"{x:8.3f}" for x in per) + f"   {per[3] / max(per[0], 1e-9):6.2f}     {sl:6.2f}               {sp:5.2f}")
