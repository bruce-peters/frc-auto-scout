"""Can the arc check credit a robot that has no box in the view (hidden behind the box the 2-D rule picks, e.g.
254 behind 3847 in a Chezy end camera)? Scored on scout truth from data/exp_arcs/log_hidden*/ (hidden_log.py).

Truth (the c24 construction, widened): candidate teams = the boxes within the link distance + the winner's
hidden teammates; exactly one has a scouted shot within +-1 s (time-valid rows, tap - 0.8 s); all are scouted.
"behind" = a hidden teammate whose fused position (x, y, 0.3 m) projects inside the 2-D winner's box (x range
widened by 25 % either side). Rules:
  2-D              today's winner
  arc boxes        lowest soft arc cost among the boxes (the box-only arc check)
  arc +behind      lowest cost among the boxes and the hidden teammates behind the winner
  arc +all hidden  lowest cost among the boxes and every hidden teammate with a position
  2-D, behind if   today's winner, unless a teammate behind it fits better by more than m (LOMO over m)

  .venv/Scripts/python data/exp_arcs/tools/eval_hidden.py [log_hidden|log_hidden_split]
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import arcs, config, field as F, scouting, tba  # noqa: E402
import eval_truth as E  # noqa: E402

EXP = config.DATA / "exp_arcs"
LOG = sys.argv[1] if len(sys.argv) > 1 else "log_hidden"


def behind(e, r, cam):
    w = next(c for c in e["cands"] if c.get("winner"))
    x0, y0, x1, y1 = w["box"]
    bw = x1 - x0
    u, v = cam.project(np.array([[r["xy_fused"][0], r["xy_fused"][1], 0.3]]))[0]
    return bool(x0 - 0.25 * bw <= u <= x1 + 0.25 * bw and y0 <= v <= y1)


def cases_from(log):
    bym = scouting.by_match(scouting.load())
    out = []
    for p in sorted((EXP / log).glob("*.pkl")):
        k = p.stem
        ev = pickle.load(open(p, "rb"))
        tl = json.loads((EXP / "base_pass1_oldmodel" / k / "result.json").read_text())["timeline"]
        info = tba.match(k)
        sc_t = {}
        for t in info.red + info.blue:
            xs = [tl["auto_start"] + x.shots - 0.8 for x in bym.get(k, {}).get(t, []) if x.time_valid]
            sc_t[t] = np.sort(np.concatenate(xs)) if xs else np.zeros(0)
        for e in ev:
            if not any(c.get("winner") for c in e["cands"]):
                continue
            cam = E.cam_of(e)
            boxes = {}  # team -> its box (the winner's own box for the winning team, else its lowest old score)
            for i, c in enumerate(e["cands"]):
                if c["src"] != "box" or c["team"] is None:
                    continue
                j = boxes.get(c["team"])
                if c.get("winner") or j is None or (not e["cands"][j].get("winner") and c["dist"] + E.TOP * c["top"]
                                                    < e["cands"][j]["dist"] + E.TOP * e["cands"][j]["top"]):
                    boxes[c["team"]] = i
            hid = {c["team"]: i for i, c in enumerate(e["cands"]) if c["src"] == "hidden" and c["team"] not in boxes}
            beh = {t: i for t, i in hid.items() if behind(e, e["cands"][i], cam)}
            teams = set(boxes) | set(hid)
            if any(not bym.get(k, {}).get(t) for t in teams):
                continue
            sh = [t for t in teams if len(sc_t.get(t, [])) and np.min(np.abs(sc_t[t] - e["t"])) <= 1.0]
            if len(sh) != 1:
                continue
            out.append({"key": k, "e": e, "boxes": boxes, "hid": hid, "beh": beh, "truth": sh[0],
                        "winner": e["winner_team"], "view": e["view"]})
    return out


def fit_case(c):
    e = c["e"]
    idx = list(c["boxes"].values()) + list(c["hid"].values())
    xys = [(e["cands"][i]["xy_fused"] or e["cands"][i]["xy_det"]) for i in idx]
    hubs = [[*F.hub_center(e["cands"][i]["alliance"]), config.HUB_OPENING_HEIGHT] for i in idx]
    fits = arcs.arc_fit(e["pts"], E.cam_of(e), xys, hubs, prior=arcs.SOFT_PRIOR)
    return {e["cands"][i]["team"]: (f["cost"] if f is not None else np.inf) for i, f in zip(idx, fits)}


def pick(c, cost, rule, m=0.0):
    if rule == "2-D":
        return c["winner"]
    pool = list(c["boxes"])
    if rule in ("arc +behind", "2-D, behind if"):
        pool += list(c["beh"])
    elif rule == "arc +all hidden":
        pool += list(c["hid"])
    if rule == "2-D, behind if":
        b = [t for t in c["beh"]]
        if not b:
            return c["winner"]
        t = min(b, key=lambda x: cost[x])
        return t if cost[t] < cost[c["winner"]] - m else c["winner"]
    return min(pool, key=lambda x: cost[x])


if __name__ == "__main__":
    cases = cases_from(LOG)
    print(f"{LOG}: {len(cases)} scout-judged box-linked launches in {len({c['key'] for c in cases})} matches; "
          f"with a hidden teammate {sum(bool(c['hid']) for c in cases)}, with one behind the winner {sum(bool(c['beh']) for c in cases)}")
    cache = EXP / f"fits_{LOG}.pkl"
    costs = pickle.load(open(cache, "rb")) if cache.exists() else None
    if costs is None or len(costs) != len(cases):
        costs = [fit_case(c) for c in cases]
        pickle.dump(costs, open(cache, "wb"))
    keys = [c["key"] for c in cases]
    grid = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 1e9]
    res = {}
    for subset, sel in (("all", lambda c: True), ("with a teammate behind the winner", lambda c: bool(c["beh"])),
                        ("end cameras, teammate behind", lambda c: bool(c["beh"]) and c["view"] != "main"),
                        ("main camera, teammate behind", lambda c: bool(c["beh"]) and c["view"] == "main")):
        ns = [n for n, c in enumerate(cases) if sel(c)]
        if not ns:
            continue
        kk = [keys[n] for n in ns]
        ref = np.array([pick(cases[n], costs[n], "2-D") == cases[n]["truth"] for n in ns], float)
        print(f"\n  {subset}: n={len(ns)} in {len(set(kk))} matches; truth is a hidden teammate in "
              f"{100 * np.mean([cases[n]['truth'] in cases[n]['hid'] for n in ns]):.0f} %, behind the winner in "
              f"{100 * np.mean([cases[n]['truth'] in cases[n]['beh'] for n in ns]):.0f} %")
        okm = {m: np.array([pick(cases[n], costs[n], "2-D, behind if", m) == cases[n]["truth"] for n in ns], float) for m in grid}
        lomo = np.zeros(len(ns))
        for k in set(kk):
            tr = np.array([x != k for x in kk])
            m = max(grid, key=lambda g: (okm[g][tr].mean(), g))
            lomo[~tr] = okm[m][~tr]
        rows = {"2-D": ref}
        for rule in ("arc boxes", "arc +behind", "arc +all hidden"):
            rows[rule] = np.array([pick(cases[n], costs[n], rule) == cases[n]["truth"] for n in ns], float)
        rows["2-D, behind if (LOMO m)"] = lomo
        out = {}
        for r, ok in rows.items():
            lo, hi = E.boot_diff(kk, ok, ref) if r != "2-D" else (0.0, 0.0)
            print(f"     {r:28s} {100 * ok.mean():5.1f} %   {100 * (ok.mean() - ref.mean()):+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")
            out[r] = {"acc": 100 * ok.mean(), "diff": 100 * (ok.mean() - ref.mean()), "ci": [100 * lo, 100 * hi]}
        print("     in-sample by m: " + ", ".join(f"{g:g}: {100 * okm[g].mean():.1f}" for g in grid))
        res[subset] = {"n": len(ns), **out}
    json.dump(res, open(EXP / f"hidden_eval_{LOG}.json", "w"), indent=1, default=float)
