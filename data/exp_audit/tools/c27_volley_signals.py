"""Cycle 27: is there a per-robot signal that grows with scouted burst size where track count doesn't?

Per flight track in the cc scans (md.tracks_all, >= 4 points): box link to a robot as in analyse step (1)
(team from dets.json), rising (initial image velocity upward), is_shot, credited (result.json), points, summed
blob area. Per scouted burst (time_valid rows, tap - 0.8 s, gaps > 1.5 s split a burst; window [first - 1,
last + 1]):
  credited    : credited launches of that robot (baseline: what attribution sees)
  shot_trk    : is_shot tracks linked to it
  (a) dur     : stream duration - credited launches clustered with gaps <= 0.5 s, sum of (span + 0.25 s)
      span    : last - first credited launch in the window (+0.25)
  (b) rise_n  : all rising tracks leaving its box (any is_shot result)
      rise_pts: their summed points (ball-frames)
      rise_mass: their summed blob area (yellow mass; merged balls stay large)
      shot_mass: summed blob area of its is_shot tracks
  (c) cnt_sole: overlay-counter balls in [first + 0.8, last + 5.5] when no other alliance robot has a credited
                launch in the burst window (NaN otherwise)
Scores: per-ball mean by burst-size class and the 20+ / 1-4 ratio (1.0 = proportional to balls), Spearman
and log-log slope vs scouted burst size. Writes data/exp_audit/c27_bursts.json (per-burst rows).
"""
import gc
import json
import pickle
import sys
from collections import defaultdict

import numpy as np

from common import A, AUDIT, DIRS, P, TL, load, match_keys, tba, time_valid
from autoscout import fuel, scouting, views

bym = scouting.by_match(scouting.load())
keys = [k for k in match_keys("v2") if k.startswith("2026cc_qm") and k in bym]
W, H = 1920, 1080
rows_out = []
for k in keys:
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
            area = float(sum(q[3] for q in tr.pts))
            trk[best[7]].append((t0, bool(vv < 0), fuel.is_shot(tr, ppm),
                                 (vname, round(t0, 3), round(u0, 1), round(v0, 1)) in credited, len(tr.pts), area))
    del md
    gc.collect()
    cred = defaultdict(list)
    for s in r["shots"]:
        if s["team"] is not None and s["toward"]:
            cred[s["team"]].append(s["t"])
    cred = {t: np.sort(v) for t, v in cred.items()}
    for al in ("red", "blue"):
        tm = info.teams(al)
        inc = A.increments_from_counts(P._match_counts(tl, al, info.hub(al) or {})) if tl.counts.get(al) else []
        it = np.array([x[0] for x in inc]); iv = np.array([x[1] for x in inc])
        for team in tm:
            T = trk.get(team, [])
            tt = np.array([x[0] for x in T])
            ct = cred.get(team, np.zeros(0))
            srows = [x for x in bym[k].get(team, []) if time_valid(x)]
            for row in srows:
                st = np.sort(tl.auto_start + row.shots - 0.8)
                if not len(st):
                    continue
                for b in np.split(st, np.where(np.diff(st) > 1.5)[0] + 1):
                    a0, a1 = b[0] - 1.0, b[-1] + 1.0
                    m = (tt >= a0) & (tt <= a1) if len(tt) else np.zeros(0, bool)
                    sel = [T[i] for i in np.where(m)[0]]
                    c = ct[(ct >= a0) & (ct <= a1)]
                    dur = 0.0
                    if len(c):
                        for cl in np.split(c, np.where(np.diff(c) > 0.5)[0] + 1):
                            dur += cl[-1] - cl[0] + 0.25
                    others = [u for u in tm if u != team]
                    sole = not any(len(cred.get(u, [])) and ((cred[u] >= a0) & (cred[u] <= a1)).any() for u in others)
                    cn = float(iv[(it >= b[0] + 0.8) & (it <= b[-1] + 5.5)].sum()) if len(it) else 0.0
                    rows_out.append({"key": k, "team": team, "n": len(b) / 1.0, "w": 1.0 / len(srows),
                                     "credited": len(c), "shot_trk": sum(1 for x in sel if x[2]),
                                     "dur": dur, "span": (c[-1] - c[0] + 0.25) if len(c) else 0.0,
                                     "rise_n": sum(1 for x in sel if x[1]), "rise_pts": sum(x[4] for x in sel if x[1]),
                                     "rise_mass": sum(x[5] for x in sel if x[1]), "shot_mass": sum(x[5] for x in sel if x[2]),
                                     "cnt_sole": cn if sole else np.nan, "sole": sole, "cnt_any": cn})
    print(k, flush=True)
json.dump(rows_out, open(AUDIT / "c27_bursts.json", "w"), default=float)
SIG = ["credited", "shot_trk", "dur", "span", "rise_n", "rise_pts", "rise_mass", "shot_mass", "cnt_sole"]
rows = rows_out
n = np.array([x["n"] for x in rows]); w = np.array([x["w"] for x in rows])
cls = np.digitize(n, [5, 10, 20])  # 0: 1-4, 1: 5-9, 2: 10-19, 3: 20+
print(f"\n{len(rows)} scouted bursts (scout rows), balls {np.sum(n * w):.0f}; sole-evidence bursts {sum(x['sole'] for x in rows)}")
print(f"{'signal':10s} " + " ".join(f"{c:>9s}" for c in ("1-4", "5-9", "10-19", "20+")) + "   20+/1-4  spearman  loglog slope")
for s in SIG:
    v = np.array([x[s] for x in rows], float)
    ok = np.isfinite(v)
    per = []
    for c in range(4):
        m = ok & (cls == c)
        per.append(np.sum(v[m] * w[m]) / max(np.sum(n[m] * w[m]), 1e-9))
    o = ok
    rk = lambda a: np.argsort(np.argsort(a))
    sp = np.corrcoef(rk(v[o]), rk(n[o]))[0, 1]
    pos = o & (v > 0)
    sl = np.polyfit(np.log(n[pos]), np.log(v[pos]), 1)[0] if pos.sum() > 10 else np.nan
    print(f"{s:10s} " + " ".join(f"{x:9.3f}" for x in per) + f"   {per[3] / max(per[0], 1e-9):7.2f}  {sp:8.2f}  {sl:8.2f}  (n={o.sum()})")
