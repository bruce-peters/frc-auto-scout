"""Prototype: absent/unknown identity state + absence-aware attribution, patched into pipeline.analyse.

Nothing in autoscout/ is edited: identity.assign and attribution.split_increments are wrapped while
CFG["on"] is True. Used by absent_eval.py.

Identity model (assign_absent): the hidden state is any map slot -> team (27 states instead of the 6
permutations). A team that is not the label of any slot (active or inactive) in an epoch is ABSENT
there and costs abs_cost[team] per epoch; two slots sharing a label is the duplicate-track case. A team
on an inactive slot is merely hidden (no cost), as before.

Dead-robot evidence (per alliance, from the first, OCR-only pass): a team with < DEAD_OWN own bumper
evidence over all segments, while every teammate has >= MATE_MIN, and whose first-pass labelled
segments carry >= FOREIGN_MIN of teammates' reads, gets the cheaper abs_cost_dead.

Attribution: w = z * (s + (PRIOR*vis + HIDDEN*(1-vis)) * (1 - absent_frac)) (+ optional extras).
"""
from __future__ import annotations

import itertools
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np  # noqa: E402

from autoscout import attribution as A, identity as I  # noqa: E402

STATES = list(itertools.product(range(3), repeat=3))
M = np.array(STATES)  # 27 x 3
IMG = np.array([[t in s for t in range(3)] for s in STATES])  # 27 x 3: team t used by state

ORIG_ASSIGN = I.assign
ORIG_SPLIT = A.split_increments

CFG = {
    "on": False,
    "abs_cost": np.inf,  # per-epoch cost of an absent team that has its own reads (inf: never)
    "abs_cost_dead": 2.0,  # ... for a dead-flagged team
    "dead_own": 0.5, "mate_min": 3.0, "foreign_min": 1.0,
    "zero": "absent",  # attribution: "none" | "absent" (PRIOR/HIDDEN 0 in absent stretches)
    "dead_hidden": False,  # dead-flagged team: HIDDEN -> 0 everywhere (hidden == absent)
    "zero_shots": False,  # also drop launches credited to a team inside its absent stretches
    "stat_w": 0.0, "stat_d0": 1.5,  # stationary gate on PRIOR/HIDDEN: window (s, 0 = off), extent (m)
    "stat_dead_only": False,
    "near_m": 0.0, "near_bonus": 0.0,  # identity: co-located concurrent segments favour a duplicate label
    "park_ext": 0.0,  # attribution: median 20 s movement extent below this (m) -> parked: PRIOR/HIDDEN 0
    "dh_min_s": None,  # dead_hidden only for dead teams absent >= this many s after relabelling
}
CTX: dict = {}


def reset(md=None):
    CTX.clear()
    CTX["md"] = md
    CTX["al"] = {}


PAIRS = [(0, 1), (0, 2), (1, 2)]
SAME = {p: M[:, p[0]] == M[:, p[1]] for p in PAIRS}  # state labels the two slots alike


def _seg_xy(seg, ts):
    p = np.asarray(seg.points, float)
    return np.interp(ts, p[:, 0], p[:, 1]), np.interp(ts, p[:, 0], p[:, 2])


def assign_absent(segs, evidence, switch_cost, abs_cost):
    """-> ({seg id: team idx}, {team idx: [(t0, t1)] absent}). abs_cost: length-3 per-epoch costs."""
    eps = I._epochs(segs)
    if not eps:
        return {}, {}
    n, S = len(eps), len(STATES)
    ac = np.asarray(abs_cost, float)
    absent_pen = np.where(IMG, 0.0, np.where(np.isfinite(ac), ac, 1e9)[None, :]).sum(1)  # 27
    score = np.full((n, S), -1e18)
    back = np.zeros((n, S), dtype=int)
    eye = np.eye(S, dtype=bool)
    near_m, near_b = CFG.get("near_m", 0.0), CFG.get("near_bonus", 0.0)

    def emit(a, b, active):
        e = -absent_pen.copy()
        for slot, seg in active.items():
            ev = evidence.get(seg.id)
            if ev is not None:
                e += np.asarray(ev, float)[M[:, slot]]
        if near_m > 0 and near_b and len(active) >= 2:
            ts = np.linspace(a, min(b, a + 5.0), 3)
            for p in PAIRS:
                if p[0] in active and p[1] in active:
                    sa, sb = active[p[0]], active[p[1]]
                    ta = np.clip(ts, sa.t0, sa.t1)
                    tb = np.clip(ts, sb.t0, sb.t1)
                    xa, ya = _seg_xy(sa, ta)
                    xb, yb = _seg_xy(sb, tb)
                    if float(np.mean(np.hypot(xa - xb, ya - yb))) < near_m:
                        e = e + near_b * SAME[p]  # co-located tracks: one robot seen twice
        return e

    score[0] = emit(*eps[0])
    for i in range(1, n):
        pa, ca = eps[i - 1][2], eps[i][2]
        cont = [sl for sl, sg in ca.items() if sl in pa and pa[sl].id == sg.id]
        ok = np.ones((S, S), bool)
        for sl in cont:
            ok &= M[:, None, sl] == M[None, :, sl]
        c = score[i - 1][:, None] - np.where(eye, 0.0, switch_cost)
        c = np.where(ok, c, -1e30)
        back[i] = np.argmax(c, axis=0)
        score[i] = c[back[i], np.arange(S)] + emit(*eps[i])
    k = int(np.argmax(score[-1]))
    path = [k]
    for i in range(n - 1, 0, -1):
        k = back[i, k]
        path.append(k)
    path.reverse()
    labels: dict = {}
    absent: dict = {0: [], 1: [], 2: []}
    for (a, b, active), k in zip(eps, path):
        for slot, seg in active.items():
            labels.setdefault(seg.id, []).append(STATES[k][slot])
        for t in range(3):
            if not IMG[k, t]:
                if absent[t] and abs(absent[t][-1][1] - a) < 1e-6:
                    absent[t][-1] = (absent[t][-1][0], b)
                else:
                    absent[t].append((a, b))
    return {sid: max(set(v), key=v.count) for sid, v in labels.items()}, absent


def _dead_flags(segs, raw):
    own = np.zeros(3)
    for s in segs:
        own += raw.get(s.id, np.zeros(3))
    first = ORIG_ASSIGN(segs, raw, 0.6)
    flags = np.zeros(3, bool)
    for t in range(3):
        mates = [u for u in range(3) if u != t]
        if own[t] < CFG["dead_own"] and all(own[u] >= CFG["mate_min"] for u in mates):
            foreign = sum(raw[sid][u] for sid, lab in first.items() if lab == t for u in mates if sid in raw)
            flags[t] = foreign >= CFG["foreign_min"]
    return flags, own


def assign_hook(segs, evidence, switch_cost=0.6):
    if not CFG["on"] or not segs:
        return ORIG_ASSIGN(segs, evidence, switch_cost)
    al = segs[0].alliance
    st = CTX["al"].setdefault(al, {"calls": 0})
    st["calls"] += 1
    if st["calls"] == 1:  # OCR-only pass: evidence = w_ocr * reads
        w_ocr = CTX.get("w_ocr", 2.0)
        raw = {k: np.asarray(v, float) / w_ocr for k, v in evidence.items()}
        st["dead"], st["own"] = _dead_flags(segs, raw)
    ac = np.where(st["dead"], CFG["abs_cost_dead"], CFG["abs_cost"])
    labels, absent = assign_absent(segs, evidence, switch_cost, ac)
    st["labels"], st["absent"], st["segs"] = labels, absent, segs
    return labels


def _overlap(iv, a, b):
    tot = 0.0
    for x, y in iv:
        lo, hi = max(a, x), min(b, y)
        if hi > lo:
            tot += hi - lo
    return tot / max(b - a, 1e-9)


def _team_ctx(team):
    md = CTX.get("md")
    if md is None:
        return None, None
    for al in ("red", "blue"):
        tm = md.info.teams(al)
        if team in tm and al in CTX["al"]:
            return CTX["al"][al], tm.index(team)
    return None, None


def _observed_pts(st, ti):
    pts = [p for s in st["segs"] if st["labels"].get(s.id) == ti for p in s.points]
    pts.sort()
    return np.array(pts) if pts else np.zeros((0, 3))


def _extent(p, a, b):
    if not len(p):
        return None
    m = (p[:, 0] >= a) & (p[:, 0] <= b)
    if m.sum() < 3:
        return None
    q = p[m, 1:]
    return float(np.hypot(*(q.max(0) - q.min(0))))


def split_hook(increments, teams, to_match_time, log=None):
    if not CFG["on"]:
        return ORIG_SPLIT(increments, teams, to_match_time, log=log)
    info = []
    for ts in teams:
        st, ti = _team_ctx(ts.team)
        if st is None or "absent" not in st:
            info.append(([], False, None, False))
            continue
        iv = st["absent"].get(ti, []) if CFG["zero"] == "absent" else []
        need_pts = CFG["stat_w"] > 0 or CFG.get("park_ext", 0) > 0
        pts = _observed_pts(st, ti) if need_pts else None
        dead = bool(st["dead"][ti])
        if CFG.get("dh_min_s") is not None:  # dead_hidden only when the relabelling confirmed it
            dead = dead and sum(b - a for a, b in st["absent"].get(ti, [])) >= CFG["dh_min_s"]
        if CFG.get("dh_max_vis") is not None:  # ... or only when the team is seen < this fraction of the match
            dead = dead and (float(ts.vis_v.mean()) if len(ts.vis_v) else 0.0) < CFG["dh_max_vis"]
        parked = False
        if CFG.get("park_ext", 0) > 0 and pts is not None and len(pts):
            ext = [e for e in (_extent(pts, a, a + 20.0) for a in np.arange(pts[0, 0], pts[-1, 0] - 20.0, 10.0))
                   if e is not None]
            parked = bool(ext) and float(np.median(ext)) < CFG["park_ext"]
            CTX.setdefault("parked", {})[ts.team] = round(float(np.median(ext)), 2) if ext else None
        if parked and CFG.get("park_mode", "all") == "all":  # dead / parked all match: no PRIOR or HIDDEN
            iv = [(-1e9, 1e9)]
        info.append((iv, dead, pts, parked and CFG.get("park_mode", "all") == "prior"))
        if CFG["zero_shots"] and iv and len(ts.shots):
            keep = np.array([_overlap(iv, t - 0.05, t + 0.05) < 0.5 for t in ts.shots])
            ts.shots, ts.shot_w = ts.shots[keep], ts.shot_w[keep]
    out = {ts.team: {} for ts in teams}
    src = CTX.setdefault("src", {})
    for t, n in increments:
        if n <= 0:
            continue
        mt = to_match_time(t - 2.0)
        per = A.period_of(mt) if mt is not None else "unknown"
        gate = A.AUTO_ZONE_GATE or per != "auto"
        w = np.zeros(len(teams))
        comp = np.zeros((len(teams), 3))
        for i, ts in enumerate(teams):
            iv, dead, pts, park_p = info[i]
            a, b = t - A.LAG_HI, t - A.LAG_LO
            z = ts.zone_frac(a, b) if gate else 1.0
            s = float((A.kernel(t - ts.shots) * ts.shot_w).sum()) if len(ts.shots) else 0.0
            vis = ts.vis_frac(a, b)
            pres = 1.0 - _overlap(iv, a, b) if iv else 1.0
            hid = 0.0 if (dead and CFG["dead_hidden"]) else A.HIDDEN * (1.0 - vis)
            g = 1.0
            if CFG["stat_w"] > 0 and pts is not None and (dead or not CFG["stat_dead_only"]):
                e = _extent(pts, t - CFG["stat_w"], t)
                if e is not None:
                    g = min(1.0, e / CFG["stat_d0"])
            comp[i] = z * np.array([s, 0.0 if park_p else A.PRIOR * vis * pres * g, hid * pres * g])
            w[i] = comp[i].sum()
        if w.sum() <= 1e-9:
            w = np.ones(len(teams))
            comp = np.ones((len(teams), 3)) / 3
        w = w / w.sum()
        for i, ts in enumerate(teams):
            out[ts.team][per] = out[ts.team].get(per, 0.0) + n * w[i]
            cs = comp[i].sum()
            src[ts.team] = src.get(ts.team, np.zeros(3)) + (n * w[i] * comp[i] / cs if cs > 0 else 0.0)
            if log is not None and mt is not None:
                log.append((round(mt, 2), ts.team, round(float(n * w[i]), 3)))
    return out


def install():
    I.assign = assign_hook
    A.split_increments = split_hook
