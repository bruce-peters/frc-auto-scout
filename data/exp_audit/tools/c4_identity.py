"""Cycle 4: identity errors the bumper reads themselves contradict.

Per track segment (dets.json rows are md.dets in order, so crop ids join to segment ids):
  * contradicted: the segment's own OCR evidence names another team clearly (>= 1.5 total, winner
    >= 70 %), but Viterbi assigned something else
  * internal swap: reads early in the segment name team A, later ones team B (each side >= 1.0
    evidence and >= 70 % for its winner) -> two robots in one segment (tracker did not cut)
Impact: credited launches (team + time inside such a segment / the wrong part of it).
Usage: c4_identity.py [v2|cc_v2|v1]
"""
import gc
import json
import pickle
import sys
from collections import defaultdict

import numpy as np

from common import AUDIT, DIRS, load, match_keys, tba
from autoscout import ocr

src = sys.argv[1] if len(sys.argv) > 1 else "v2"
tot = defaultdict(float)
examples = []
BAD = {}
for k in match_keys(src):
    r = load(k, src); info = tba.match(k)
    rows = json.loads((DIRS[src] / k / "dets.json").read_text())
    with open(DIRS[src] / k / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cache = getattr(md, "ocr_cache", {}) or {}
    if len(md.dets) != len(rows):
        print("skip", k, "dets mismatch"); continue
    seg_reads = defaultdict(list)  # sid -> [(t, ev vector over alliance teams)]
    seg_team, seg_al = {}, {}
    for d, row in zip(md.dets, rows):
        sid, team = row[8], row[7]
        if sid is None:
            continue
        seg_team[sid], seg_al[sid] = team, row[6]
        if d.crop_id is not None and d.crop_id in cache and cache[d.crop_id]:
            ev = ocr.team_evidence(cache[d.crop_id], info.teams(row[6]))
            if ev.sum() > 0:
                seg_reads[sid].append((d.t, ev))
    del md
    gc.collect()
    seg_span = {}
    for al in ("red", "blue"):
        for sid, e in r["identity"][al].items():
            seg_span[int(sid)] = (e["t0"], e["t1"], e["team"])
    bad = {}  # sid -> (t_from, t_to) wrong interval
    n_seg = n_read_seg = 0
    for sid, rd in seg_reads.items():
        teams = info.teams(seg_al[sid])
        rd.sort(key=lambda x: x[0])
        E = np.array([x[1] for x in rd]); T = np.array([x[0] for x in rd])
        n_read_seg += 1
        tot_ev = E.sum(0)
        assigned = seg_team[sid]
        ai = teams.index(assigned) if assigned in teams else None
        win = int(np.argmax(tot_ev))
        t0, t1 = T[0], T[-1]
        if tot_ev.sum() >= 1.5 and tot_ev[win] / tot_ev.sum() >= 0.7 and ai is not None and win != ai:
            bad[sid] = ("contradicted", seg_span.get(sid, (t0, t1))[0], seg_span.get(sid, (t0, t1))[1], teams[win])
            continue
        # internal swap: best split point
        cs = np.cumsum(E, axis=0)
        best = None
        for i in range(1, len(E)):
            a, b = cs[i - 1], cs[-1] - cs[i - 1]
            if a.sum() >= 1.0 and b.sum() >= 1.0:
                wa, wb = int(np.argmax(a)), int(np.argmax(b))
                if wa != wb and a[wa] / a.sum() >= 0.7 and b[wb] / b.sum() >= 0.7:
                    sc = a[wa] + b[wb]
                    if best is None or sc > best[0]:
                        best = (sc, i, wa, wb)
        if best is not None:
            _, i, wa, wb = best
            # the part that disagrees with the assigned team is wrong
            if ai == wa:
                bad[sid] = ("swap_late", (T[i - 1] + T[i]) / 2, seg_span.get(sid, (t0, t1))[1], teams[wb])
            elif ai == wb:
                bad[sid] = ("swap_early", seg_span.get(sid, (t0, t1))[0], (T[i - 1] + T[i]) / 2, teams[wa])
            else:
                bad[sid] = ("swap_neither", seg_span.get(sid, (t0, t1))[0], seg_span.get(sid, (t0, t1))[1], None)
    n_seg = len(seg_span)
    # credited launches inside bad intervals of their own team's segment
    cred = [s for s in r["shots"] if s["team"] is not None and s["toward"]]
    hit = defaultdict(int)
    for s in cred:
        for sid, (kind, a, b, should) in bad.items():
            if seg_team.get(sid) == s["team"] and a - 0.3 <= s["t"] <= b + 0.3:
                hit[kind] += 1
                if len(examples) < 60:
                    examples.append({"key": k, "t": s["t"], "view": s["view"], "how": s["how"], "credited": s["team"],
                                     "reads_say": should, "kind": kind, "sid": sid})
                break
    BAD[k] = [(int(sid), kind, float(a), float(b), seg_team.get(sid), should) for sid, (kind, a, b, should) in bad.items()]
    dur_bad = sum(b - a for kind, a, b, _ in bad.values())
    tot["matches"] += 1; tot["segments"] += n_seg; tot["segments_with_reads"] += n_read_seg
    for kind, *_ in bad.values():
        tot[f"seg_{kind}"] += 1
    tot["bad_seconds"] += dur_bad
    tot["credited"] += len(cred)
    for kk, v in hit.items():
        tot[f"launch_{kk}"] += v
    print(f"{k}: segs {n_seg} with reads {n_read_seg}; contradicted {sum(1 for x in bad.values() if x[0] == 'contradicted')}, "
          f"internal swaps {sum(1 for x in bad.values() if x[0].startswith('swap'))}; credited launches in bad parts "
          f"{sum(hit.values())} / {len(cred)}", flush=True)
print("\nTOTAL", dict(tot))
nb = sum(v for kk, v in tot.items() if kk.startswith("launch_"))
print(f"credited launches on the wrong team per the segment's own reads: {nb} / {tot['credited']:.0f} "
      f"({100 * nb / max(tot['credited'], 1):.1f} %)")
json.dump({"tot": tot, "examples": examples, "bad": BAD}, open(AUDIT / f"c4_identity_{src}.json", "w"), indent=1, default=float)
