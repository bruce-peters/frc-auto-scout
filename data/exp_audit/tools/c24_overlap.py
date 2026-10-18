"""Cycle 24: box-overlap launch credit. When several same-view robot boxes meet the box-link test at the
launch point, pipeline.analyse picks the lowest score = dist + TOP_TIEBREAK*|v0 - y0|/bh. How often is that
the wrong robot, and does another rule do better?

Per credited box-linked launch (output/): candidates = boxes at the nearest det time (<= 0.2 s) with
dist <= 0.6 and a team. Ambiguous = >= 2 candidate teams. Ground truth (cc quals with scouting): exactly one
candidate team has a scouted shot (time_valid rows, tap - 0.8 s) within +-1.0 s.
Rules for ambiguous launches:
  tie       : current pick
  zone      : candidates in their own zone at launch (teleop) first, then score
  recent    : candidate whose team has the most UNambiguous credited launches within +-5 s, then score
  dir       : candidates whose alliance hub the arc heads toward first, then score
  front/back: larger box (nearer the camera) / smaller box
  split     : conf split equally across candidate teams
  zone+dir  : zone, then dir, then score
Re-split -> scout share corr (primary) and fuel OPR.
"""
import copy
import gc
import json
import pickle
from collections import Counter, defaultdict

import numpy as np

from common import AUDIT, DIRS, F, P, TL, load, match_keys, path_video_times, resplit, scout_metrics, share_metrics, tba, time_valid
from autoscout import scouting, views

bym = scouting.by_match(scouting.load())
R, INFO, AMB = {}, {}, {}
stats = Counter()
judged = defaultdict(Counter)


def launch_point(s):
    p = np.array([q[1:] for q in s["pts"][:4]]); t = np.array([q[0] for q in s["pts"][:4]])
    A_ = np.vstack([t - t[0], np.ones(len(t))]).T
    vu = np.linalg.lstsq(A_, p[:, 0], rcond=None)[0][0]; vv = np.linalg.lstsq(A_, p[:, 1], rcond=None)[0][0]
    return s["u"] - vu * 2 / 60, s["v"] - vv * 2 / 60


for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k); tl = TL(r)
    R[k], INFO[k] = r, info
    rows = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    with open(DIRS["v2"] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cams, prod = md.cams, md.prod
    del md
    gc.collect()
    vl = {v.name: v for v in views.PRODUCTIONS[prod]}
    by_vt = defaultdict(lambda: defaultdict(list))
    for d in rows:
        by_vt[d[1]][d[0]].append(d)
    tk = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    paths = {t: path_video_times(r["paths"].get(str(t), []), tl) for t in info.red + info.blue}
    al_of = {t: ("red" if t in info.red else "blue") for t in info.red + info.blue}
    amb = []
    for si, s in enumerate(r["shots"]):
        if s["team"] is None or s["how"] != "box" or s["view"] not in tk:
            continue
        stats["box credited"] += 1
        ts = tk[s["view"]]
        j = int(np.argmin(np.abs(ts - s["t"])))
        if abs(ts[j] - s["t"]) > 0.2:
            continue
        ox, oy = vl[s["view"]].offset(1920, 1080)
        lu, lv = launch_point(s)
        cands = {}
        for d in by_vt[s["view"]][ts[j]]:
            if d[7] is None:
                continue
            x0, y0, x1, y1 = d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy
            bw, bh = x1 - x0, y1 - y0
            dx = max(x0 - 0.25 * bw - lu, 0, lu - x1 - 0.25 * bw); dy = max(y0 - 0.9 * bh - lv, 0, lv - y1)
            dist = np.hypot(dx, dy) / max(bh, 1)
            if dist <= 0.6:
                sc_ = dist + P.TOP_TIEBREAK * abs(s["v"] - y0) / max(bh, 1)
                if d[7] not in cands or sc_ < cands[d[7]]["score"]:
                    cands[d[7]] = {"score": sc_, "area": bw * bh}
        if len(cands) < 2:
            continue
        cam = cams.get(s["view"]); c = cam.at(s["t"]) if cam is not None else None
        ppm = P._px_per_m(c, s["u"], s["v"])
        mt = tl.match_time(s["t"])
        for t, cd in cands.items():
            al = al_of[t]
            p = P._track_position(paths[t], s["t"]) if len(paths[t]) else None
            cd["zone"] = bool(p is not None and F.in_alliance_zone(al, p[0]))
            if c is not None:
                hc = F.hub_center(al)
                hu, hv = c.project(np.array([[hc[0], hc[1], 1.9]]))[0]
                d0 = np.hypot(s["pts"][0][1] - hu, s["pts"][0][2] - hv); d1 = np.hypot(s["pts"][-1][1] - hu, s["pts"][-1][2] - hv)
                cd["toward"] = bool(d1 < d0 - 0.3 * ppm)
            else:
                cd["toward"] = True
        amb.append({"i": si, "t": s["t"], "auto": mt is not None and mt < 20, "pick": s["team"], "cands": cands,
                    "cross": len({al_of[t] for t in cands}) > 1})
    # recent unambiguous launches per team
    amb_idx = {a["i"] for a in amb}
    unamb = defaultdict(list)
    for si, s in enumerate(r["shots"]):
        if s["team"] is not None and s["toward"] and si not in amb_idx:
            unamb[s["team"]].append(s["t"])
    for a in amb:
        for t, cd in a["cands"].items():
            cd["recent"] = sum(1 for x in unamb.get(t, []) if abs(x - a["t"]) <= 5.0)
        stats["ambiguous"] += 1
        stats["ambiguous cross-alliance" if a["cross"] else "ambiguous same-alliance"] += 1
    # scout ground truth
    if k in bym:
        sc_t = {t: np.sort(np.concatenate([tl.auto_start + x.shots - 0.8 for x in bym[k].get(t, []) if time_valid(x)] or [np.zeros(0)]))
                for t in info.red + info.blue}
        for a in amb:
            shooters = [t for t in a["cands"] if len(sc_t.get(t, [])) and np.min(np.abs(sc_t[t] - a["t"])) <= 1.0]
            unscouted = [t for t in a["cands"] if not bym[k].get(t)]
            if len(shooters) == 1 and not unscouted:
                a["truth"] = shooters[0]
    AMB[k] = amb
    print(k, len(amb), flush=True)

RULES = ["tie", "zone", "recent", "dir", "front", "back", "zone+dir"]


def choose(a, rule):
    c = a["cands"]
    order = sorted(c, key=lambda t: c[t]["score"])
    if rule == "tie":
        return a["pick"]
    if rule == "zone":
        z = [t for t in order if c[t]["zone"]] if not a["auto"] else []
        return (z or order)[0]
    if rule == "recent":
        return max(order, key=lambda t: (c[t]["recent"], -c[t]["score"]))
    if rule == "dir":
        d = [t for t in order if c[t]["toward"]]
        return (d or order)[0]
    if rule == "front":
        return max(order, key=lambda t: c[t]["area"])
    if rule == "back":
        return min(order, key=lambda t: c[t]["area"])
    if rule == "zone+dir":
        z = [t for t in order if c[t]["zone"]] if not a["auto"] else order
        d = [t for t in (z or order) if c[t]["toward"]]
        return (d or z or order)[0]


print("\n", dict(stats))
truth = [a for k in AMB for a in AMB[k] if "truth" in a]
print(f"ambiguous launches with a scout-identified shooter: {len(truth)} "
      f"(cross-alliance {sum(a['cross'] for a in truth)}, same-alliance {sum(not a['cross'] for a in truth)})")
for rule in RULES:
    ok = [choose(a, rule) == a["truth"] for a in truth]
    ok_s = [choose(a, rule) == a["truth"] for a in truth if not a["cross"]]
    ok_c = [choose(a, rule) == a["truth"] for a in truth if a["cross"]]
    print(f"   {rule:9s} correct {100 * np.mean(ok):5.1f} %  (same-alliance {100 * np.mean(ok_s) if ok_s else float('nan'):5.1f} %, "
          f"cross {100 * np.mean(ok_c) if ok_c else float('nan'):5.1f} %)")
print(f"   chance (1 / #candidates) {100 * np.mean([1 / len(a['cands']) for a in truth]):.1f} %")

res = {}
for rule in RULES + ["split"]:
    pm = {}
    for k in R:
        r2 = copy.deepcopy(R[k])
        extra = []
        for a in AMB[k]:
            s = r2["shots"][a["i"]]
            if rule == "split":
                n = len(a["cands"]); conf = s["conf"] / n
                s["conf"] = conf if s["team"] in a["cands"] else s["conf"]
                for t in a["cands"]:
                    if t != s["team"]:
                        e = dict(s); e["team"] = t; e["conf"] = conf; extra.append(e)
            else:
                s["team"] = choose(a, rule)
        r2["shots"] += extra
        pm[k] = resplit(r2, INFO[k])
    res[rule] = pm
    print(f"{rule:9s}", share_metrics(pm), scout_metrics(pm), flush=True)
json.dump({"stats": dict(stats), "truth_n": len(truth)}, open(AUDIT / "c24_overlap.json", "w"))
json.dump([{**{kk: vv for kk, vv in a.items() if kk != "cands"}, "cands": {str(t): v for t, v in a["cands"].items()}, "key": k}
           for k in AMB for a in AMB[k] if "truth" in a], open(AUDIT / "c24_truth_cases.json", "w"), indent=0, default=float)
