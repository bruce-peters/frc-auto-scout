"""Cycle 17: where do stream / volley shots get lost? Stage-by-stage per scouted burst.

For every flight track in output/ cc scans (md.tracks_all, all tracks >= 4 points): box link exactly like
pipeline.analyse step (1) (launch point from velocity0, expanded box, TOP_TIEBREAK, bd <= 0.6) using the
dets.json team of that box; is_shot with its failure reason; SHOT_GATE (_box_shot_ok) failure reason;
whether it was finally credited (result.json shots). Scouted bursts (time_valid rows, tap - 0.8 s, gaps
> 1.5 s split a burst) of each robot, by burst size: tracks linked to that robot per scouted ball at each
stage, rejection reasons, and track features (points, duration, blob area, speed).
Usage: c17_stages.py [match ...]   (default: every cc qual in output/ with scouting)
"""
import gc
import json
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

from common import AUDIT, DIRS, P, TL, load, match_keys, tba, time_valid
from autoscout import fuel, scouting, views

bym = scouting.by_match(scouting.load())
keys = sys.argv[1:] or [k for k in match_keys("v2") if k.startswith("2026cc_qm") and k in bym]
W, H = 1920, 1080


def shot_reason(tr, ppm):
    if len(tr.pts) < 4:
        return "short"
    vu, vv = tr.velocity0()
    if np.hypot(vu, vv) / max(ppm, 1e-3) < 3.5:
        return "slow"
    if vv > -0.25 * abs(vu) and vv > -1.0 * ppm:
        return "not rising"
    if tr.fit()["res"] >= 6.0:
        return "not smooth"
    return None


def gate_reason(tr, box, view_h):
    x0, y0, x1, y1 = box
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    p = np.array([(q[1], q[2]) for q in tr.pts])
    u0, v0 = p[0]
    if (y0 - v0) / bh > P.LAUNCH_ABOVE_MAX:
        return "starts too high"
    if not (x0 - P.SIDE_MARGIN * bw <= u0 <= x1 + P.SIDE_MARGIN * bw):
        return "beside box"
    if (y0 - p[:, 1].min()) / bh < P.RISE_ABOVE_MIN and p[:, 1].min() > P.EXIT_TOP * view_h:
        return "no rise"
    if ((p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)).mean() > P.INBOX_MAX:
        return "inside box"
    return None


stage = defaultdict(lambda: np.zeros(5))  # (who, size class) -> balls, linked, is_shot, gated-pass, credited
reasons = defaultdict(Counter)            # (who, size class) -> Counter of failure reasons
feats = defaultdict(list)                 # (who, size class) -> [(npts, dur, area_med, speed)]
for k in keys:
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    rows = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    with open(DIRS["v2"] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    vl = {v.name: v for v in views.PRODUCTIONS[md.prod]}
    cams = md.cams
    by_vt = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_vt[row[1]][row[0]].append(row)
    tk = {v: np.array(sorted(d)) for v, d in by_vt.items()}
    credited = {(s["view"], round(s["t"], 3), round(s["u"], 1), round(s["v"], 1)) for s in r["shots"] if s["team"] is not None}
    trk = defaultdict(list)  # team -> [(t0, linked flags...)]
    for vname, tracks in md.tracks_all.items():
        if vname not in tk:
            continue
        ox, oy = vl[vname].offset(W, H)
        _, vy0, _, vy1 = vl[vname].scaled(W, H)
        cam_track = cams.get(vname)
        for tr in tracks:
            if len(tr.pts) < 4:
                continue
            t0, u0, v0 = tr.pts[0][0], tr.pts[0][1], tr.pts[0][2]
            ts = tk[vname]
            j = int(np.argmin(np.abs(ts - t0)))
            if abs(ts[j] - t0) > 0.2:
                continue
            vu, vv = tr.velocity0()
            lu, lv = u0 - vu * 2 / 60, v0 - vv * 2 / 60
            best, bd, bs, bbox = None, 1e9, 1e9, None
            for d in by_vt[vname][ts[j]]:
                x0, y0, x1, y1 = d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy
                bw, bh = x1 - x0, y1 - y0
                dx = max(x0 - 0.25 * bw - lu, 0, lu - x1 - 0.25 * bw); dy = max(y0 - 0.9 * bh - lv, 0, lv - y1)
                dist = np.hypot(dx, dy) / max(bh, 1)
                sc_ = dist + P.TOP_TIEBREAK * abs(v0 - y0) / max(bh, 1)
                if sc_ < bs:
                    best, bd, bs, bbox = d, dist, sc_, (x0, y0, x1, y1)
            if best is None or bd > 0.6 or best[7] is None:
                continue
            cam = cam_track.at(t0) if cam_track is not None else None
            ppm = P._px_per_m(cam, u0, v0)
            sr = shot_reason(tr, ppm)
            gr = gate_reason(tr, bbox, vy1 - vy0) if sr is None else None
            cr = (vname, round(t0, 3), round(u0, 1), round(v0, 1)) in credited
            a = np.array([q[3] for q in tr.pts])
            trk[best[7]].append((t0, sr, gr, cr, len(tr.pts), tr.pts[-1][0] - t0, float(np.median(a)),
                                 float(np.hypot(vu, vv) / max(ppm, 1e-3))))
    del md
    gc.collect()
    for al in ("red", "blue"):
        for team in info.teams(al):
            T = sorted(trk.get(team, []), key=lambda x: x[0])
            tt = np.array([x[0] for x in T])
            srows = [x for x in bym[k].get(team, []) if time_valid(x)]
            if not srows:
                continue
            who = "5026" if team == 5026 else "others"
            for row in srows:
                st = np.sort(tl.auto_start + row.shots - 0.8)
                if not len(st):
                    continue
                cuts = np.where(np.diff(st) > 1.5)[0]
                bursts = np.split(st, cuts + 1)
                for b in bursts:
                    n = len(b)
                    cls = "1-4" if n < 5 else ("5-9" if n < 10 else ("10-19" if n < 20 else "20+"))
                    m = (tt >= b[0] - 1.0) & (tt <= b[-1] + 1.0) if len(tt) else np.zeros(0, bool)
                    sel = [T[i] for i in np.where(m)[0]]
                    w = 1.0 / len(srows)
                    for key in ((who, cls), (who, "all"), ("all", cls)):
                        stage[key] += w * np.array([n, len(sel), sum(1 for x in sel if x[1] is None),
                                                    sum(1 for x in sel if x[1] is None and x[2] is None), sum(1 for x in sel if x[3])])
                        for x in sel:
                            reasons[key][x[1] or (x[2] or "passed")] += w
                            feats[key].append(x[4:])
    print(k, "done", flush=True)
print("\nper scouted ball: tracks starting on the robot's box (linked) -> pass is_shot -> pass gate -> credited")
order = ["1-4", "5-9", "10-19", "20+", "all"]
for who in ("all", "others", "5026"):
    for cls in order:
        key = (who, cls)
        if key not in stage:
            continue
        b, l, s, g, c = stage[key]
        f = np.array(feats[key]) if feats[key] else np.zeros((0, 4))
        rs = reasons[key]; tot = sum(rs.values())
        top = ", ".join(f"{a} {100 * v / tot:.0f}%" for a, v in rs.most_common(6))
        print(f"  {who:6s} burst {cls:5s} balls {b:6.0f} | linked {l / b:.2f} is_shot {s / b:.2f} gate {g / b:.2f} credited {c / b:.2f} per ball"
              f" | pass rates is_shot {s / max(l, 1e-9):.2f} gate {g / max(s, 1e-9):.2f}"
              + (f" | track pts {np.median(f[:, 0]):.0f} dur {np.median(f[:, 1]):.2f}s area {np.median(f[:, 2]):.0f} speed {np.median(f[:, 3]):.1f}"
                 if len(f) else "") + f"\n             fates: {top}")
json.dump({"stage": {f"{a}|{b}": v.tolist() for (a, b), v in stage.items()},
           "reasons": {f"{a}|{b}": dict(v) for (a, b), v in reasons.items()}}, open(AUDIT / "c17_stages.json", "w"), indent=1)
