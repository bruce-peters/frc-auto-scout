"""Cycle 4b: for segments whose own reads name another team: (1) is the read team simultaneously held by
another segment, and is that one physically on top of this one (a duplicate / ghost track)?
(2) re-credit launches inside those intervals to the read team -> share metrics."""
import copy, json, sys
from collections import Counter
import numpy as np
from common import AUDIT, P, TL, load, path_video_times, resplit, scout_metrics, share_metrics, tba
src = sys.argv[1] if len(sys.argv) > 1 else "v2"
B = json.load(open(AUDIT / f"c4_identity_{src}.json"))["bad"]
cat = Counter(); dists = []
base, alt = {}, {}
alt_dup, alt_far = {}, {}
ex = []
for k, bad in B.items():
    r = load(k, src); info = tba.match(k); tl = TL(r)
    paths = {t: path_video_times(r["paths"].get(str(t), []), tl) for t in info.red + info.blue}
    r2 = copy.deepcopy(r); r3 = copy.deepcopy(r); r4 = copy.deepcopy(r)
    for sid, kind, a, b, assigned, should in bad:
        if should is None or assigned is None:
            continue
        # is `should` tracked during [a, b]? how far from `assigned`?
        pb = paths.get(should); pa = paths.get(assigned)
        ts = np.arange(a, b + 1e-6, 0.5)
        d = []
        for t in ts:
            x = P._track_position(pb, t) if len(pb) else None
            y = P._track_position(pa, t) if len(pa) else None
            if x is not None and y is not None and x[2] and y[2]:
                d.append(np.hypot(x[0] - y[0], x[1] - y[1]))
        if not d:
            cat[f"{kind}: read team not seen at the same time"] += 1
        else:
            md_ = float(np.median(d)); dists.append(md_)
            cat[f"{kind}: read team tracked elsewhere, median {'< 1 m (duplicate?)' if md_ < 1.0 else '>= 1 m'}"] += 1
        dup = bool(d) and float(np.median(d)) < 1.0
        if dup and len(ex) < 12 and b - a > 2:
            ex.append((k, sid, round(a, 1), round(b, 1), assigned, should, round(float(np.median(d)), 2)))
        for s, s2, s3, s4 in zip(r["shots"], r2["shots"], r3["shots"], r4["shots"]):
            if s["team"] == assigned and s["toward"] and a - 0.3 <= s["t"] <= b + 0.3:
                s2["team"] = should
                (s3 if dup else s4)["team"] = should
    base[k] = resplit(r, info); alt[k] = resplit(r2, info); alt_dup[k] = resplit(r3, info); alt_far[k] = resplit(r4, info)
for c, v in sorted(cat.items()):
    print(f"  {c:70s} {v}")
if dists:
    print("  distance read-team vs assigned-team track (m): quantiles 10/50/90", np.round(np.quantile(dists, [.1, .5, .9]), 2))
print("base  ", share_metrics(base), scout_metrics(base))
print("relink", share_metrics(alt), scout_metrics(alt))
print("relink duplicates only", share_metrics(alt_dup), scout_metrics(alt_dup))
print("relink far only       ", share_metrics(alt_far), scout_metrics(alt_far))
print("duplicate examples (match, sid, t0, t1, assigned, reads say, median dist m):")
for e in ex:
    print("  ", e)
