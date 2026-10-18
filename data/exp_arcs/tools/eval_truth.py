"""Pass 2: pick accuracy of the 3-D arc check on the failure finder's scout-judged truth set
(data/exp_audit/c24_truth_cases.json: 1,084 box-linked launches with >= 2 candidate robots where exactly one
candidate had a scouted shot within +-1 s).

Each case is matched to its launch in data/exp_arcs/log/<match>.pkl by launch time, box area and the old score
(dist + TOP_TIEBREAK * top), so the boxes carry the case's own team labels. Rules compared on the same boxes:
  old       dist + 0.5 top                          (the c24 pick; 62.5 %)
  old+zone  ... cross-alliance teleop: own-zone robot first
  current   dist + 0.5 top + LINK_X_TIE * xc, zone rule (today's pipeline)
  arc       lowest arc_fit cost alone (zone rule first)
  cur+arc   current + W * (cost - min cost) / SIGMA2
Also a second truth set built the same way from the current labels (fresh), for the matches with scouting.

Pick accuracy ignores the launch gate. The outcome table scores what the pipeline would credit: the pick among
ALL close candidates (zone rule); a pick whose box fails pipeline._box_shot_ok is dropped (counted wrong):
  gate on winner  the gate checks the re-ranked winner (a plain extra score term)
  gate-aware      the term may only move the pick to a box that passes the gate
  gate on 2-D     shot / no shot decided on today's 2-D winner, the term only chooses the robot (recommended)

  .venv/Scripts/python data/exp_arcs/tools/eval_truth.py [--variants ...]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "exp_audit" / "tools"))
from autoscout import arcs, config, field as F, scouting, tba  # noqa: E402
from autoscout.field import Camera  # noqa: E402

EXP = config.DATA / "exp_arcs"
TOP, XT = 0.5, 1.0


def load_log(k, log_dir="log"):
    p = EXP / log_dir / f"{k}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)


def cam_of(e):
    c = e["cam"]
    return Camera(c["K"], c["dist"], c["rvec"], c["tvec"], tuple(c["size"]), 0.0, 0)


def match_case(case, events):
    """The logged launch and, per c24 candidate team, the logged candidate index; None if no match."""
    best = None
    for ei, e in enumerate(events):
        if abs(e["t"] - case["t"]) > 1e-4:
            continue
        idx, cost = {}, 0.0
        used = set()
        for team, cd in case["cands"].items():
            j_best, d_best = None, 1e9
            for j, c in enumerate(e["cands"]):
                if j in used:
                    continue
                da = abs(c["area"] - cd["area"]) / max(cd["area"], 1.0)
                ds = abs(c["dist"] + TOP * c["top"] - cd["score"])
                if da < 0.02 and ds < 0.05 and da + ds < d_best:
                    j_best, d_best = j, da + ds
            if j_best is None:
                break
            idx[int(team)] = j_best
            used.add(j_best)
            cost += d_best
        else:
            if best is None or cost < best[2]:
                best = (ei, idx, cost)
    return best


def zone_ok(c):
    return c["xy_det"] is not None and F.in_alliance_zone(c["alliance"], c["xy_det"][0])


def pick(e, idx, rule, fits=None, W=0.0, S2=100.0, zone=True):
    """Team picked among the case's candidates (idx: team -> logged candidate index)."""
    teams = list(idx)
    cs = [e["cands"][idx[t]] for t in teams]
    if zone and len({c["alliance"] for c in cs}) > 1 and e["mt"] is not None and e["mt"] >= config.AUTO_LEN:
        inz = [i for i, c in enumerate(cs) if zone_ok(c)]
        keep = inz or list(range(len(cs)))
    else:
        keep = list(range(len(cs)))
    if rule == "old":
        s = [cs[i]["dist"] + TOP * cs[i]["top"] for i in keep]
    elif rule == "current":
        s = [cs[i]["dist"] + TOP * cs[i]["top"] + XT * cs[i]["xc"] for i in keep]
    elif rule in ("arc", "cur+arc"):
        cost = np.array([fits[idx[teams[i]]]["cost"] if fits[idx[teams[i]]] is not None else np.nan for i in keep])
        if not np.isfinite(cost).any():
            cost = np.zeros(len(keep))
        cost = np.where(np.isfinite(cost), cost, np.nanmax(cost))
        rel = (cost - cost.min()) / S2
        if rule == "arc":
            s = list(rel)
        else:
            s = [cs[i]["dist"] + TOP * cs[i]["top"] + XT * cs[i]["xc"] + W * rel[n] for n, i in enumerate(keep)]
    return teams[keep[int(np.argmin(s))]]


def fit_event(e, pos="fused", launch_z=arcs.LAUNCH_Z, max_pts=None, prior=None):
    cam = cam_of(e)
    xys, hubs = [], []
    for c in e["cands"]:
        xy = c["xy_fused"] if pos == "fused" else None
        xys.append(xy if xy is not None else c["xy_det"])
        hubs.append([*F.hub_center(c["alliance"]), config.HUB_OPENING_HEIGHT])
    return arcs.arc_fit(e["pts"], cam, xys, hubs, launch_z=launch_z, max_pts=max_pts, prior=prior)


def boot_diff(keys, a, b, n=2000, seed=0):
    """Paired bootstrap over matches of mean(a) - mean(b) (per-case 0/1 arrays grouped by match key)."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for k, x, y in zip(keys, a, b):
        by[k].append((x, y))
    ks = list(by)
    out = []
    for _ in range(n):
        s = rng.choice(len(ks), len(ks))
        xs = [v for i in s for v in by[ks[i]]]
        A = np.array(xs)
        out.append(A[:, 0].mean() - A[:, 1].mean())
    return np.percentile(out, [2.5, 97.5])


def c24_cases():
    """[(case, event, idx)] for the c24 truth cases that match a logged launch."""
    truth = json.loads((config.DATA / "exp_audit" / "c24_truth_cases.json").read_text())
    logs, matched = {}, []
    for c in truth:
        if c["key"] not in logs:
            logs[c["key"]] = load_log(c["key"])
        ev = logs[c["key"]]
        if ev is None:
            continue
        m = match_case(c, ev)
        if m is not None:
            matched.append((c, ev[m[0]], m[1]))
    return truth, matched


def current_pick_index(e):
    """Index of the logged candidate today's pipeline links (zone rule over all close candidates), or None."""
    cs = e["cands"]
    keep = list(range(len(cs)))
    if len({c["alliance"] for c in cs}) > 1 and e["mt"] is not None and e["mt"] >= config.AUTO_LEN:
        inz = [i for i in keep if zone_ok(cs[i])]
        keep = inz or keep
    return min(keep, key=lambda i: cs[i]["score"])


def fresh_cases(log_dir="log"):
    """The c24 construction on the current labels: credited box-linked launches (today's pick has a team and
    passes the launch gate) with >= 2 candidate teams, exactly one of which the scouts logged shooting within
    +-1 s (time-valid rows, tap - 0.8 s), every candidate team scouted."""
    bym = scouting.by_match(scouting.load())
    out = []
    for k in sorted(bym):
        ev = load_log(k, log_dir)
        rp = EXP / "base_pass1_oldmodel" / k / "result.json"  # only its timeline is used
        if ev is None or not rp.exists():
            continue
        tl = json.loads(rp.read_text())["timeline"]
        info = tba.match(k)
        sc_t = {}
        for t in info.red + info.blue:
            xs = [tl["auto_start"] + x.shots - 0.8 for x in bym[k].get(t, []) if x.time_valid]
            sc_t[t] = np.sort(np.concatenate(xs)) if xs else np.zeros(0)
        for e in ev:
            j = current_pick_index(e)
            if e["cands"][j]["team"] is None or not e["cands"][j]["gate_ok"]:
                continue
            idx = {}
            for i, c in enumerate(e["cands"]):
                if c["team"] is None:
                    continue
                s_old = c["dist"] + TOP * c["top"]
                if c["team"] not in idx or s_old < e["cands"][idx[c["team"]]]["dist"] + TOP * e["cands"][idx[c["team"]]]["top"]:
                    idx[c["team"]] = i
            if len(idx) < 2 or any(not bym[k].get(t) for t in idx):
                continue
            sh = [t for t in idx if len(sc_t.get(t, [])) and np.min(np.abs(sc_t[t] - e["t"])) <= 1.0]
            if len(sh) != 1:
                continue
            case = {"key": k, "t": e["t"], "truth": sh[0], "pick": e["cands"][j]["team"],
                    "cross": len({e["cands"][i]["alliance"] for i in idx.values()}) > 1}
            out.append((case, e, idx))
    return out


def accuracy(cases, picks):
    ok = np.array([p == c["truth"] for (c, e, i), p in zip(cases, picks)], float)
    cr = np.array([c["cross"] for c, e, i in cases])
    return ok, 100 * ok.mean(), 100 * ok[~cr].mean() if (~cr).any() else np.nan, 100 * ok[cr].mean() if cr.any() else np.nan


def lomo_lambda(cases, fits, grid):
    """Leave-one-match-out choice of lambda = W / SIGMA2 for cur+arc: per held-out match, the lambda with the
    best accuracy on the other matches (ties -> smallest). Returns the held-out picks and the chosen lambdas."""
    keys = np.array([c["key"] for c, e, i in cases])
    ok = {lam: np.array([pick(e, i, "cur+arc", f, W=lam, S2=1.0) == c["truth"] for (c, e, i), f in zip(cases, fits)])
          for lam in grid}
    picks, chosen = [None] * len(cases), {}
    for k in np.unique(keys):
        tr = keys != k
        lam = max(grid, key=lambda g: (ok[g][tr].mean(), -g))
        chosen[k] = lam
        for n in np.where(~tr)[0]:
            c, e, i = cases[n]
            picks[n] = pick(e, i, "cur+arc", fits[n], W=lam, S2=1.0)
    return picks, chosen, ok


def report(name, cases, fit_sets, lam_grid):
    keys = [c["key"] for c, e, i in cases]
    print(f"\n=== {name}: {len(cases)} cases in {len(set(keys))} matches "
          f"(same-alliance {sum(not c['cross'] for c, e, i in cases)}, cross {sum(c['cross'] for c, e, i in cases)})")
    rows = {}
    base = {"old": [pick(e, i, "old", zone=False) for c, e, i in cases],
            "old+zone": [pick(e, i, "old") for c, e, i in cases],
            "current": [pick(e, i, "current") for c, e, i in cases]}
    for r, p in base.items():
        rows[r] = accuracy(cases, p)
    for vname, fits in fit_sets.items():
        rows[f"arc[{vname}]"] = accuracy(cases, [pick(e, i, "arc", f) for (c, e, i), f in zip(cases, fits)])
        picks, chosen, ok = lomo_lambda(cases, fits, lam_grid)
        rows[f"current+arc[{vname}] LOMO"] = accuracy(cases, picks)
        best = max(lam_grid, key=lambda g: ok[g].mean())
        rows[f"current+arc[{vname}] in-sample best lam={best:g}"] = accuracy(cases, [
            pick(e, i, "cur+arc", f, W=best, S2=1.0) for (c, e, i), f in zip(cases, fits)])
        lam_counts = {g: list(chosen.values()).count(g) for g in lam_grid if g in chosen.values()}
        print(f"   [{vname}] LOMO lambda picks: {lam_counts}; in-sample accuracy by lambda: "
              + ", ".join(f"{g:g}: {100 * ok[g].mean():.1f}" for g in lam_grid))
    cur_ok = rows["current"][0]
    print(f"   {'rule':44s} {'all':>6s} {'same':>6s} {'cross':>6s}   diff vs current [95 % CI, matches]")
    out = {}
    for r, (ok, a_, s_, c_) in rows.items():
        lo, hi = boot_diff(keys, ok, cur_ok) if r != "current" else (0.0, 0.0)
        print(f"   {r:44s} {a_:6.1f} {s_:6.1f} {c_:6.1f}   {a_ - rows['current'][1]:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")
        out[r] = {"all": a_, "same": s_, "cross": c_, "diff": a_ - rows["current"][1], "ci": [100 * lo, 100 * hi]}
    return out


def breakdown(cases, fits, lam):
    """current vs cur+arc(lam) by view and arc duration (same-alliance only)."""
    print(f"\n   breakdown, same-alliance, current vs current+arc (lambda {lam:g}) vs arc alone:")
    groups = defaultdict(list)
    for n, (c, e, i) in enumerate(cases):
        if c["cross"]:
            continue
        dur = e["pts"][-1, 0] - e["pts"][0, 0]
        groups[("main" if e["view"] == "main" else "end", "dur<0.2" if dur < 0.2 else ("dur<0.5" if dur < 0.5 else "dur>=0.5"))].append(n)
    for g in sorted(groups):
        ns = groups[g]
        cur = np.mean([pick(cases[n][1], cases[n][2], "current") == cases[n][0]["truth"] for n in ns])
        ca = np.mean([pick(cases[n][1], cases[n][2], "cur+arc", fits[n], W=lam, S2=1.0) == cases[n][0]["truth"] for n in ns])
        ar = np.mean([pick(cases[n][1], cases[n][2], "arc", fits[n]) == cases[n][0]["truth"] for n in ns])
        print(f"     {g[0]:5s} {g[1]:9s} n={len(ns):4d}  current {100 * cur:5.1f}  current+arc {100 * ca:5.1f}  arc {100 * ar:5.1f}")


def outcome(e, idx, rule, f=None, lam=0.3, gate="winner"):
    """Team the pipeline credits for this launch (None = dropped by the launch gate, "other" = a box outside the
    case's candidate set). rule "current" or "cur+arc"; gate "winner" / "aware" / "2d" (see the module doc)."""
    cs = e["cands"]
    keep = list(range(len(cs)))
    if len({c["alliance"] for c in cs}) > 1 and e["mt"] is not None and e["mt"] >= config.AUTO_LEN:
        inz = [i for i in keep if zone_ok(cs[i])]
        keep = inz or keep
    s0 = np.array([cs[i]["score"] for i in keep])
    base = keep[int(np.argmin(s0))]
    s = s0
    if rule == "cur+arc":
        cost = np.array([f[i]["cost"] if f[i] is not None else np.nan for i in keep])
        if np.isfinite(cost).any():
            cost = np.where(np.isfinite(cost), cost, np.nanmax(cost))
            s = s0 + lam * (cost - cost.min())
        if gate == "aware":
            s = s0 if not cs[base]["gate_ok"] else s + np.array([0.0 if cs[i]["gate_ok"] else 1e3 for i in keep])
    j = keep[int(np.argmin(s))]
    if not cs[base if gate == "2d" else j]["gate_ok"]:
        return None
    inv = {v: t for t, v in idx.items()}
    return inv.get(j, "other")


def outcome_report(name, cases, fit_sets, lams):
    keys = [c["key"] for c, e, i in cases]
    cr = np.array([c["cross"] for c, e, i in cases])
    print(f"\n   {name}, pipeline outcome (dropped by the launch gate = wrong):")
    print(f"   {'rule':40s} {'all':>6s} {'same':>6s} {'cross':>6s} {'dropped':>8s}   diff vs current [95 % CI, matches]")
    rows = [("current", "current", None, None, "winner")]
    for v, lam in lams.items():
        rows += [(f"current+arc[{v}] lam {lam:g}, gate on winner", "cur+arc", v, lam, "winner"),
                 (f"current+arc[{v}] lam {lam:g}, gate-aware", "cur+arc", v, lam, "aware"),
                 (f"current+arc[{v}] lam {lam:g}, gate on 2-D", "cur+arc", v, lam, "2d")]
    out, ref = {}, None
    for label, rule, v, lam, gate in rows:
        fits = fit_sets[v] if v else [None] * len(cases)
        o = [outcome(e, i, rule, f, lam=lam or 0.0, gate=gate) for (c, e, i), f in zip(cases, fits)]
        ok = np.array([x == c["truth"] for (c, e, i), x in zip(cases, o)], float)
        drop = 100 * np.mean([x is None for x in o])
        ref = ok if ref is None else ref
        lo, hi = boot_diff(keys, ok, ref) if rule != "current" else (0.0, 0.0)
        d = 100 * (ok.mean() - ref.mean())
        print(f"   {label:40s} {100 * ok.mean():6.1f} {100 * ok[~cr].mean():6.1f} {100 * ok[cr].mean():6.1f} {drop:7.1f} %   "
              f"{d:+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]")
        out[label] = {"all": 100 * ok.mean(), "same": 100 * ok[~cr].mean(), "cross": 100 * ok[cr].mean(), "dropped": drop,
                      "diff": d, "ci": [100 * lo, 100 * hi]}
    return out


VARIANTS = {"hard": {}, "hard_det": {"pos": "det"}, "soft": {"prior": arcs.SOFT_PRIOR},
            "soft_det": {"pos": "det", "prior": arcs.SOFT_PRIOR}}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="soft,hard")
    ap.add_argument("--log", default=None, help="score only a fresh set built from data/exp_arcs/<log> (e.g. log_split)")
    a = ap.parse_args()
    vnames = a.variants.split(",")
    lam_grid = [0.0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0]
    if a.log:
        cases = fresh_cases(a.log)
        cache = EXP / f"fits_{a.log}.pkl"
        fit_sets = pickle.load(open(cache, "rb")) if cache.exists() else {}
        for v in vnames:
            if v not in fit_sets:
                fit_sets[v] = [fit_event(e, **VARIANTS[v]) for c, e, i in cases]
        pickle.dump(fit_sets, open(cache, "wb"))
        name = f"fresh truth set from {a.log}"
        res = {name: report(name, cases, {v: fit_sets[v] for v in vnames}, lam_grid)}
        breakdown(cases, fit_sets[vnames[0]], 0.3)
        lams = {"soft": 0.3, "hard": 0.01}
        res[name + " / outcome"] = outcome_report(name, cases, fit_sets, {v: lams[v] for v in vnames if v in lams})
        json.dump(res, open(EXP / f"truth_eval_{a.log}.json", "w"), indent=1, default=float)
        sys.exit(0)
    truth, matched = c24_cases()
    print(f"c24 truth cases {len(truth)}, matched to a logged launch {len(matched)}")
    agree = np.mean([pick(e, idx, "old", zone=False) == c["pick"] for c, e, idx in matched])
    print(f"old rule reproduces the c24 pick in {100 * agree:.1f} % of matched cases")
    fresh = fresh_cases()
    lam_grid = [0.0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1.0, 3.0]
    res = {}
    for name, cases in (("c24 truth set (labels as of cycle 24)", matched), ("fresh truth set (current labels)", fresh)):
        cache = EXP / f"fits_{'c24' if name.startswith('c24') else 'fresh'}.pkl"
        fit_sets = pickle.load(open(cache, "rb")) if cache.exists() else {}
        for v in vnames:
            if v not in fit_sets:
                fit_sets[v] = [fit_event(e, **VARIANTS[v]) for c, e, i in cases]
        pickle.dump(fit_sets, open(cache, "wb"))
        res[name] = report(name, cases, {v: fit_sets[v] for v in vnames}, lam_grid)
        breakdown(cases, fit_sets[vnames[0]], 0.3)
        lams = {"soft": 0.3, "hard": 0.01}  # the leave-one-match-out picks in report() (every fold, both sets)
        res[name + " / outcome"] = outcome_report(name, cases, fit_sets, {v: lams[v] for v in vnames if v in lams})
    json.dump(res, open(EXP / "truth_eval.json", "w"), indent=1, default=float)
