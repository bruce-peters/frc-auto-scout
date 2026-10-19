"""The two-view launch point as a soft term in today's 2-D score (like the arc term of data/exp_arcs):

    score_i = dist_i + TOP * top_i + LINK_X_TIE * xc_i + lam * d2_i,   d2_i = Mahalanobis^2 of candidate i's fused
    position from the launch point (launch covariance + SIGMA_ROBOT^2)

on the c24 and fresh truth sets (zone rule first, as eval_truth.pick "current"); only launches with an accepted
two-view fit get the term. lam by leave-one-match-out over a grid. Paired bootstrap vs today over matches.

  .venv/Scripts/python data/exp_multiview/tools/eval_soft.py [pairs|pairs_lm]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import eval_mv as EM  # noqa: E402
import eval_truth as E  # noqa: E402
from autoscout import config, field as F, triangulate as T  # noqa: E402

if len(sys.argv) > 1:
    C.PAIRS_DIR = sys.argv[1]
GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
RULES = ["any fit", "rms<=15", "rms<=10", "rms<=6", "norm<=2 (main 8, end 4 px)", "any, mutual strict"]


def d2_terms(e, idx, c):
    """{team: d2} for the case's candidates, or None without an accepted fit."""
    if c is None or c["launch"] is None:
        return None
    L = c["launch"]
    S = L["cov"] + T.SIGMA_ROBOT ** 2 * np.eye(2)
    Si = np.linalg.inv(S)
    out = {}
    for t, j in idx.items():
        xy = e["cands"][j]["xy_fused"] or e["cands"][j]["xy_det"]
        if xy is None:
            out[t] = np.nan
            continue
        r = np.asarray(xy[:2]) - L["xy"]
        out[t] = float(r @ Si @ r)
    return out


def pick_soft(e, idx, d2, lam):
    teams = list(idx)
    cs = [e["cands"][idx[t]] for t in teams]
    if len({c["alliance"] for c in cs}) > 1 and e["mt"] is not None and e["mt"] >= config.AUTO_LEN:
        inz = [i for i, c in enumerate(cs) if E.zone_ok(c)]
        keep = inz or list(range(len(cs)))
    else:
        keep = list(range(len(cs)))
    s = np.array([cs[i]["dist"] + E.TOP * cs[i]["top"] + E.XT * cs[i]["xc"] for i in keep])
    if d2 is not None and lam > 0:
        dd = np.array([d2[teams[i]] for i in keep])
        if np.isfinite(dd).any():
            dd = np.where(np.isfinite(dd), dd, np.nanmax(dd))
            s = s + lam * (dd - dd.min())
    return teams[keep[int(np.argmin(s))]]


def run(name, cases):
    keys = np.array([c["key"] for c, e, i in cases])
    out = {}
    print(f"\n=== {name}: {len(cases)} cases")
    ref = np.array([E.pick(e, i, "current") == c["truth"] for c, e, i in cases], float)
    for rn in RULES:
        d2s = []
        for c, e, i in cases:
            a = EM.anchor_for(c["key"], e["view"], e["pts"])
            cand = C.best_candidate(a, EM.RULES[rn], EM._load(c["key"])[1]) if a is not None else None
            d2s.append(d2_terms(e, i, cand))
        cov = np.array([d is not None for d in d2s])
        ok = {lam: np.array([pick_soft(e, i, d, lam) == c["truth"] for (c, e, i), d in zip(cases, d2s)], float) for lam in GRID}
        lomo = np.zeros(len(cases))
        chosen = []
        for k in np.unique(keys):
            tr = keys != k
            lam = max(GRID, key=lambda g: (ok[g][tr].mean(), -g))
            chosen.append(lam)
            lomo[~tr] = ok[lam][~tr]
        lo, hi = E.boot_diff(list(keys), lomo, ref)
        insample = ", ".join(f"{g:g}: {100 * ok[g].mean():.1f}" for g in GRID)
        print(f"   {rn:28s} cover {100 * cov.mean():5.1f}%  LOMO {100 * lomo.mean():5.1f} vs today {100 * ref.mean():5.1f}"
              f"  {100 * (lomo.mean() - ref.mean()):+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]  lam picks {sorted(set(chosen))}"
              f"\n        in-sample by lam: {insample}")
        out[rn] = {"cover": float(cov.mean()), "lomo": float(lomo.mean()), "today": float(ref.mean()), "ci": [float(lo), float(hi)],
                   "in_sample": {str(g): float(ok[g].mean()) for g in GRID}}
    return out


if __name__ == "__main__":
    truth, matched = E.c24_cases()
    res = {"c24": run("c24", matched), "fresh": run("fresh", E.fresh_cases())}
    json.dump(res, open(C.EXP / f"eval_soft_{C.PAIRS_DIR}.json", "w"), indent=1)
