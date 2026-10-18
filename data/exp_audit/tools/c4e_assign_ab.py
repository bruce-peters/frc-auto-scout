"""Cycle 4e: A/B of identity.assign variants through the full analyse() (in memory, nothing written
under output/). Metrics: held-out identity agreement (scripts/identity_holdout.py definition), share corr
vs fuel OPR, share corr vs scouting.
  prod     : identity.assign as is
  perseg   : each segment's evidence divided by the number of epochs it spans
  repair   : prod labels, then swap labels between a strongly-read contradicted segment and the
             concurrent segment(s) holding its read team when that raises the OCR agreement and keeps
             labels exclusive in every epoch
Usage: c4e_assign_ab.py SRC match...   (writes data/exp_audit/c4e_<tag>.json)
"""
import gc
import json
import pickle
import sys
import time
from collections import Counter

import numpy as np

from common import AUDIT, DIRS, P, scout_metrics, share_metrics, tba
from autoscout import identity, ocr

src, keys = sys.argv[1], sys.argv[2:]
ORIG = identity.assign
MISS = [0]
_orig_read = ocr.read_numbers


def _read(img):
    MISS[0] += 1
    return []


ocr.read_numbers = _read  # every read should be cached already; count misses


def assign_perseg(segs, evidence, switch_cost=0.6):
    eps = identity._epochs(segs)
    n = Counter()
    for _, _, act in eps:
        for sg in act.values():
            n[sg.id] += 1
    return ORIG(segs, {k: v / max(1, n[k]) for k, v in evidence.items()}, switch_cost)


CUR_OCR = {}


def assign_repair(segs, evidence, switch_cost=0.6):
    lab = dict(ORIG(segs, evidence, switch_cost))
    ocr_ev = CUR_OCR  # raw OCR per segment id for this alliance (set by the wrapper below)
    eps = identity._epochs(segs)
    by_id = {s.id: s for s in segs}

    def feasible(l):
        for _, _, act in eps:
            ls = [l.get(sg.id) for sg in act.values()]
            ls = [x for x in ls if x is not None]
            if len(ls) != len(set(ls)):
                return False
        return True

    def agree(l, ids):
        return sum(float(ocr_ev[i][l[i]]) for i in ids if i in ocr_ev and l.get(i) is not None)

    for _ in range(3):
        changed = False
        for s in segs:
            e = ocr_ev.get(s.id)
            if e is None or e.sum() < 1.5 or e.max() / e.sum() < 0.7:
                continue
            want = int(np.argmax(e))
            have = lab.get(s.id)
            if have is None or have == want:
                continue
            # concurrent holders of `want`
            hold = [o.id for o in segs if o.id != s.id and lab.get(o.id) == want and o.t0 < s.t1 and o.t1 > s.t0]
            new = dict(lab)
            new[s.id] = want
            for o in hold:
                new[o] = have
            ids = [s.id] + hold
            if agree(new, ids) > agree(lab, ids) and feasible(new):
                lab = new
                changed = True
        if not changed:
            break
    return lab


from autoscout import robots

ORIG_FINISH = robots.FieldTracker.finish


def split_finish(max_len):
    """FieldTracker.finish that cuts every segment longer than max_len s into <= max_len s pieces
    (same slot), so identity.assign may re-decide the label mid-track when the evidence says so."""
    def fin(self):
        out = []
        for sg in ORIG_FINISH(self):
            if sg.t1 - sg.t0 <= max_len:
                out.append(sg)
                continue
            n = int(np.ceil((sg.t1 - sg.t0) / max_len))
            edges = sg.t0 + (sg.t1 - sg.t0) * np.arange(n + 1) / n
            for a, b in zip(edges[:-1], edges[1:]):
                last = b >= sg.t1 - 1e-9
                pts = [p for p in sg.points if a <= p[0] < b or (last and p[0] == b)]
                dts = [d for d in sg.dets if a <= d.t < b or (last and d.t == b)]
                if pts:
                    out.append(robots.Segment(0, sg.alliance, sg.slot, pts[0][0], pts[-1][0], pts, dts))
        for i, sg in enumerate(out):
            sg.id = i
        return out
    return fin


VARIANTS = {"prod": (ORIG, None), "perseg": (assign_perseg, None), "split10": (ORIG, 10.0),
            "split10_perseg": (assign_perseg, 10.0)}


def run(md, opts, spec):
    fn, split = spec
    # identity.assign is called per alliance; give assign_repair the raw OCR of that alliance's segments
    calls = []

    def wrapper(segs, evidence, switch_cost=0.6):
        global CUR_OCR
        CUR_OCR = {}
        if fn is assign_repair:
            al = segs[0].alliance if segs else None
            CUR_OCR = {k: v for k, v in RAW.get(al, {}).items()}
        return fn(segs, evidence, switch_cost)

    P.identity.assign = wrapper
    if split:
        robots.FieldTracker.finish = split_finish(split)
    try:
        return P.analyse(md, opts)
    finally:
        P.identity.assign = ORIG
        robots.FieldTracker.finish = ORIG_FINISH


RAW = {}
res = {v: {"agree": 0.0, "mass": 0.0, "fuel": {}} for v in VARIANTS}
t_start = time.time()
for k in keys:
    with open(DIRS[src] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    info = md.info
    # raw OCR evidence per segment comes from a production pass (ident_debug 'ocr')
    base = P.analyse(md, P.Options())
    RAW = {al: {int(s): np.array(e["ocr"]) for s, e in base["identity"][al].items()} for al in ("red", "blue")}
    rh_prod = {hk: P.analyse(md, P.Options(ocr_holdout=hk)) for hk in (0, 1)}
    for name, fn in VARIANTS.items():
        RAW = {al: {int(s): np.array(e["ocr"]) for s, e in base["identity"][al].items()} for al in ("red", "blue")}
        r = base if name == "prod" else run(md, P.Options(), fn)
        res[name]["fuel"][k] = {t: r["teams"][t]["fuel"] for t in info.red + info.blue}
        for hk in (0, 1):
            # holdout: identity from half the reads (repair also only sees that half)
            RAW = {al: {int(s): np.array(e["ocr"]) for s, e in rh_prod[hk]["identity"][al].items()} for al in ("red", "blue")}
            rh = rh_prod[hk] if name == "prod" else run(md, P.Options(ocr_holdout=hk), fn)
            for al in ("red", "blue"):
                teams = info.teams(al)
                for s in rh["identity"][al].values():
                    if s["team"] is None:
                        continue
                    res[name]["agree"] += s["held"][teams.index(s["team"])]
                    res[name]["mass"] += sum(s["held"])
    del md
    gc.collect()
    print(f"{k} done ({time.time() - t_start:.0f}s, OCR misses {MISS[0]})", flush=True)
out = {}
for name, v in res.items():
    pm = {k: {int(t): f for t, f in d.items()} for k, d in v["fuel"].items()}
    out[name] = {"heldout_identity": v["agree"] / max(v["mass"], 1e-9), **share_metrics(pm), **(scout_metrics(pm) or {})}
    print(name, {a: round(b, 4) if isinstance(b, float) else b for a, b in out[name].items()})
tag = f"{src}_{len(keys)}"
(AUDIT / f"c4e_{tag}.json").write_text(json.dumps({"keys": keys, "metrics": out,
                                                    "fuel": {n: v["fuel"] for n, v in res.items()}}, indent=1, default=float))
