"""cc_qm24 70-74 s, main view, with pipeline.SPLIT_TRACKS on (in memory): which robot does the 3-D arc check pick
for the split arcs that 2-D linking gives to 3847 / 5507 or drops at the launch gate?

Logs every split shot track in the window with every box in the view at that time (any link distance), then
fits arcs.arc_fit (soft priors) for each box with a team, and compares with the scouts' shots in the window.

  .venv/Scripts/python -u data/exp_arcs/tools/split_diag.py [match] [t0] [t1] [view] [--gate-trace]

Also fits every robot that has no box near the arc in this view but a fused position from its track (a robot hidden
behind another one), as candidate source "track".
"""
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import arcs, config, pipeline, scouting, tba  # noqa: E402
import arc_hook  # noqa: E402

key = sys.argv[1] if len(sys.argv) > 1 else "2026cc_qm24"
T0, T1 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (70.0, 74.0)
VIEW = sys.argv[4] if len(sys.argv) > 4 else "main"
GT = "--gate-trace" in sys.argv


class WindowLog:
    def __init__(self):
        self.ev = []
        self._segs = None

    def __call__(self, md, tr, cam, cands, t0, vname, det_seg, segs, team_of_seg, tl, ppm, view_h, lu, lv, u0, v0, vu, vv):
        if vname != VIEW or not (T0 <= t0 <= T1):
            return cands
        if self._segs is None or self._segs[0] is not segs:
            self._segs = (segs, arc_hook._seg_index(segs))
        rows = []
        for score, dist, d, box in cands:
            sid = det_seg.get(id(d))
            fx = arc_hook.fused_xy(self._segs[1].get(sid), t0) if sid is not None else None
            g = {}
            for gt in (False, True):
                pipeline.GATE_TRACE = gt
                g[gt] = bool(pipeline._box_shot_ok(tr, box, view_h))
            pipeline.GATE_TRACE = GT
            rows.append({"score": float(score), "dist": float(dist), "box": box, "team": team_of_seg.get(sid) if sid is not None else None,
                         "alliance": d.alliance, "xy": fx[:2] if fx else d.xy, "gate": g[False], "gate_trace": g[True],
                         "top_px": float(box[1] - v0), "bh": float(box[3] - box[1]), "src": "box"})
        # robots with a fused position but no box near the arc in this view (hidden or not detected here)
        have = {r["team"] for r in rows if r["team"] is not None and r["dist"] <= 2.0}
        for s in segs:
            tm = team_of_seg.get(s.id)
            if tm is None or tm in have or not (s.t0 - 0.3 <= t0 <= s.t1 + 0.3):
                continue
            fx = arc_hook.fused_xy(s, t0)
            if fx is None or fx[2] > 0.6:
                continue
            al = "red" if tm in md.info.red else "blue"
            top = cam.project(np.array([[fx[0], fx[1], 0.55]]))[0] if cam is not None else (np.nan, np.nan)
            rows.append({"score": np.nan, "dist": np.nan, "box": None, "team": tm, "alliance": al, "xy": fx[:2],
                         "gate": False, "gate_trace": False, "top_px": float(top[1] - v0), "bh": np.nan,
                         "src": "track", "du": float(top[0] - u0)})
            have.add(tm)
        self.ev.append({"t": float(t0), "pts": np.asarray(tr.pts), "cam": cam, "rows": rows})
        return cands


arc_hook.build()
with open(config.OUTPUT / key / "scan.pkl", "rb") as fh:
    raw = fh.read()
md = pickle.loads(raw)
del raw
pipeline.load_dino(md, config.OUTPUT / key)
pipeline.SPLIT_TRACKS = True
pipeline.GATE_TRACE = GT
log = WindowLog()
pipeline._ARC_HOOK = log
res = pipeline.analyse_arc(md, pipeline.Options())
pipeline._ARC_HOOK = None

info = tba.match(key)
a0 = res["timeline"]["auto_start"]
bym = scouting.by_match(scouting.load()).get(key, {})
print(f"{key} {VIEW} {T0:g}-{T1:g} s video, SPLIT_TRACKS on")
print("scouted shots in the window (launch = auto start + tap - 0.8 s), mean over scouts:")
for t in info.red + info.blue:
    rows = [r for r in bym.get(t, []) if r.time_valid]
    if rows:
        n = np.mean([np.sum((a0 + r.shots - 0.8 >= T0) & (a0 + r.shots - 0.8 <= T1)) for r in rows])
        if n:
            print(f"   {t}: {n:.1f}")
print("credited in the pipeline (shots list):", dict(Counter((s["team"], s["how"]) for s in res["shots"]
                                                              if s["view"] == VIEW and T0 <= s["t"] <= T1)))
pick2d, pick_arc, pick_arc_close, why = Counter(), Counter(), Counter(), Counter()
for e in log.ev:
    rows = [r for r in e["rows"] if r["team"] is not None]
    brs = [r for r in e["rows"] if r["src"] == "box"]
    best = min(brs, key=lambda r: r["score"]) if brs else None
    if best is None or best["dist"] > 0.6:
        why["no box within link distance"] += 1
    elif not (best["gate_trace"] if GT else best["gate"]):
        why[f"2-D winner {best['team']} fails gate" + (" (passes traced)" if best["gate_trace"] else "")] += 1
    else:
        pick2d[best["team"]] += 1
    near = [r for r in rows if r["src"] == "box" and r["dist"] <= 2.0 and r["xy"] is not None]
    if len(near) >= 1:
        hubs = [arc_hook.hub_xyz(r["alliance"]) for r in near]
        fits = arcs.arc_fit(e["pts"], e["cam"], [r["xy"] for r in near], hubs, prior=arcs.SOFT_PRIOR)
        c = [f["cost"] for f in fits]
        j = int(np.argmin(c))
        pick_arc[near[j]["team"]] += 1
        cl = [k for k, r in enumerate(near) if r["dist"] <= 0.6]
        if cl:
            pick_arc_close[near[min(cl, key=lambda k: c[k])]["team"]] += 1
        e["fits"] = list(zip([r["team"] for r in near], [r["dist"] for r in near], c))
hid = Counter()
for e in log.ev:
    rows = [r for r in e["rows"] if r["team"] is not None and r["xy"] is not None]
    box_close = [r for r in rows if r["src"] == "box" and r["dist"] <= 0.6]
    best = min(box_close, key=lambda r: r["score"]) if box_close else None
    if best is None or not (best["gate_trace"] if GT else best["gate"]):
        continue
    same = [r for r in rows if r["alliance"] == best["alliance"] and (r["src"] == "track" or r["dist"] <= 2.0)]
    if not any(r["src"] == "track" for r in same):
        continue
    fits = arcs.arc_fit(e["pts"], e["cam"], [r["xy"] for r in same], [arc_hook.hub_xyz(x["alliance"]) for x in same],
                        prior=arcs.SOFT_PRIOR)
    j = int(np.argmin([f["cost"] for f in fits]))
    hid[(best["team"], same[j]["team"], same[j]["src"])] += 1
print("\nlinked launches where a teammate has no box near the arc: (2-D winner, arc pick over winner + teammates, its source):")
for kk, n in sorted(hid.items(), key=lambda x: -x[1]):
    print("   ", kk, n)
print(f"\n{len(log.ev)} split shot tracks in the window with boxes in the view")
print("2-D link (winner within link distance and passing the gate):", dict(pick2d))
print("not linked by box:", dict(why))
print("arc fit, lowest cost among boxes with a team within dist 2.0:", dict(pick_arc))
print("arc fit among boxes within the link distance (0.6):", dict(pick_arc_close))
print("\nper track: t, first-sighting px above each box top / box height, dist, gate, gate_trace, arc cost")
for e in log.ev[:40]:
    s = []
    for r in sorted([r for r in e["rows"] if r["team"] is not None and r["src"] == "box" and r["dist"] <= 2.0], key=lambda r: r["dist"]):
        cost = next((c for t, d, c in e.get("fits", []) if t == r["team"] and abs(d - r["dist"]) < 1e-9), np.nan)
        s.append(f"{r['team']}: up {r['top_px']:.0f}/{r['bh']:.0f}px d {r['dist']:.2f} g{int(r['gate'])}{int(r['gate_trace'])} c {cost:.0f}")
    print(f"  {e['t']:.2f} n={len(e['pts'])}  " + " | ".join(s))
