"""Cycle 24d: 'shooter signature' - where on its own box a robot's arcs start, learned from its unambiguous
box-linked launches (per team, all its matches at the event), used to pick between overlapping boxes.

Offset of the first arc point relative to the box: ((u0 - cx) / bw, (v0 - y0) / bh). Per team: mean and
std (shrunk towards the pooled distribution). Ambiguous case: pick the candidate with the highest Gaussian
likelihood of the observed offset under its own signature. Accuracy on the scout-judged cases.
"""
import json
from collections import defaultdict

import numpy as np

from common import AUDIT, DIRS, load, match_keys, tba
from autoscout import views

T = json.load(open(AUDIT / "c24_truth_cases.json"))
amb_keys = defaultdict(set)
for a in T:
    amb_keys[a["key"]].add(a["i"])
# all ambiguous indices (any case) are unknown here; treat box launches with only one candidate within 0.6 as unambiguous


def launch_pt(s):
    p = np.array([q[1:] for q in s["pts"][:4]]); t = np.array([q[0] for q in s["pts"][:4]])
    A_ = np.vstack([t - t[0], np.ones(len(t))]).T
    return s["u"] - np.linalg.lstsq(A_, p[:, 0], rcond=None)[0][0] * 2 / 60, s["v"] - np.linalg.lstsq(A_, p[:, 1], rcond=None)[0][0] * 2 / 60


sig = defaultdict(list)          # (event, team) -> offsets from unambiguous launches
case_cands = {}                  # (key, i) -> {team: offset}
events = {a["key"].split("_")[0] for a in T}
for k in match_keys("v2"):
    ev = k.split("_")[0]
    if ev not in events:
        continue
    r = load(k, "v2")
    prod = "chezy" if r["timeline"]["layout"] == "chezy" else "first"
    vl = {v.name: v for v in views.PRODUCTIONS[prod]}
    rows = json.loads((DIRS["v2"] / k / "dets.json").read_text())
    by_vt = defaultdict(lambda: defaultdict(list))
    for d in rows:
        by_vt[d[1]][d[0]].append(d)
    tk = {v: np.array(sorted(x)) for v, x in by_vt.items()}
    for si, s in enumerate(r["shots"]):
        if s["how"] != "box" or s["team"] is None or s["view"] not in tk:
            continue
        ts = tk[s["view"]]; j = int(np.argmin(np.abs(ts - s["t"])))
        if abs(ts[j] - s["t"]) > 0.2:
            continue
        ox, oy = vl[s["view"]].offset(1920, 1080)
        lu, lv = launch_pt(s)
        offs = {}
        for d in by_vt[s["view"]][ts[j]]:
            if d[7] is None:
                continue
            x0, y0, x1, y1 = d[2] - ox, d[3] - oy, d[4] - ox, d[5] - oy
            bw, bh = x1 - x0, y1 - y0
            dx = max(x0 - 0.25 * bw - lu, 0, lu - x1 - 0.25 * bw); dy = max(y0 - 0.9 * bh - lv, 0, lv - y1)
            if np.hypot(dx, dy) / max(bh, 1) <= 0.6:
                offs[d[7]] = ((s["u"] - (x0 + x1) / 2) / max(bw, 1), (s["v"] - y0) / max(bh, 1))
        if len(offs) == 1 and s["team"] in offs:
            sig[(ev, s["team"])].append(offs[s["team"]])
        if si in amb_keys.get(k, set()):
            case_cands[(k, si)] = offs
allo = np.array([o for v in sig.values() for o in v])
mu0, sd0 = allo.mean(0), allo.std(0)
print(f"unambiguous launches {len(allo)}; pooled offset mean {np.round(mu0, 2)} sd {np.round(sd0, 2)}")
spread = [np.array(v).std(0).mean() for v in sig.values() if len(v) >= 10]
print(f"per-team offset sd (teams with >= 10): median {np.median(spread):.2f} vs pooled {sd0.mean():.2f}")


def loglik(ev, team, o, k0=10.0):
    x = np.array(sig.get((ev, team), [])).reshape(-1, 2)
    n = len(x)
    mu = (x.sum(0) + k0 * mu0) / (n + k0) if n else mu0
    var = ((((x - mu) ** 2).sum(0) if n else 0) + k0 * sd0 ** 2) / (n + k0)
    return float(-0.5 * (((np.array(o) - mu) ** 2) / var + np.log(var)).sum())


ok = defaultdict(list)
for a in T:
    offs = case_cands.get((a["key"], a["i"]))
    if not offs or len(offs) < 2:
        continue
    ev = a["key"].split("_")[0]
    pk = max(offs, key=lambda t: loglik(ev, t, offs[t]))
    key = "cross" if a["cross"] else "same"
    for rule, p in (("tie", a["pick"]), ("signature", pk)):
        ok[(rule, key)].append(int(p) == int(a["truth"])); ok[(rule, "all")].append(int(p) == int(a["truth"]))
for rule in ("tie", "signature"):
    print(f"   {rule:10s} all {100 * np.mean(ok[(rule, 'all')]):5.1f} % (n={len(ok[(rule, 'all')])})  same-alliance "
          f"{100 * np.mean(ok[(rule, 'same')]):5.1f} %  cross {100 * np.mean(ok[(rule, 'cross')]):5.1f} %")
