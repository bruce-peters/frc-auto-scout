"""Cycle 4d: why does identity.assign label segments against their own strong bumper reads?

Rebuild each alliance's segments (id, slot, t0, t1) and OCR evidence from result['identity'] and re-run
identity.assign with OCR evidence only (w_ocr * ocr, no appearance):
  * as is (evidence counted once per epoch the segment spans)
  * evidence divided by the number of epochs the segment spans (counted once per segment)
  * evidence weighted by segment duration / epoch duration (per-second evidence)
Compare with the stored (production: OCR + appearance) labels: how many strongly-read segments
(>= 1.5, winner >= 70 %) each labelling contradicts.
Usage: c4d_viterbi.py [v2]
"""
import sys
from collections import Counter
from dataclasses import dataclass

import numpy as np

from common import load, match_keys, tba
from autoscout import identity

src = sys.argv[1] if len(sys.argv) > 1 else "v2"


@dataclass
class Seg:
    id: int
    slot: int
    t0: float
    t1: float


def strong(ev):
    ev = np.asarray(ev)
    return ev.sum() >= 1.5 and ev.max() / ev.sum() >= 0.7


cnt = Counter()
for k in match_keys(src):
    r = load(k, src); info = tba.match(k)
    for al in ("red", "blue"):
        teams = info.teams(al)
        ide = r["identity"][al]
        segs = [Seg(int(s), e["slot"], e["t0"], e["t1"]) for s, e in ide.items()]
        ocr = {int(s): np.array(e["ocr"]) for s, e in ide.items()}
        prod = {int(s): (teams.index(e["team"]) if e["team"] in teams else None) for s, e in ide.items()}
        eps = identity._epochs(segs)
        n_ep = Counter()
        for a, b, act in eps:
            for sl, sg in act.items():
                n_ep[sg.id] += 1
        variants = {
            "production (OCR+appearance)": prod,
            "OCR only, per-epoch (as coded)": identity.assign(segs, {i: 2.0 * v for i, v in ocr.items()}, 0.6),
            "OCR only, once per segment": identity.assign(segs, {i: 2.0 * v / max(1, n_ep[i]) for i, v in ocr.items()}, 0.6),
        }
        for name, lab in variants.items():
            for s in segs:
                if strong(ocr[s.id]):
                    cnt[(name, "strong")] += 1
                    if lab.get(s.id) is not None and lab[s.id] != int(np.argmax(ocr[s.id])):
                        cnt[(name, "contradicted")] += 1
                        cnt[(name, "contradicted_s")] += s.t1 - s.t0
            cnt[(name, "label_changes_vs_prod")] += sum(1 for s in segs if lab.get(s.id) != prod.get(s.id))
        cnt["segments"] += len(segs)
        cnt["epochs"] += len(eps)
        cnt["epochs_per_segment"] += sum(n_ep.values())
print(f"{len(match_keys(src))} matches, {cnt['segments']} segments, {cnt['epochs']} epochs; a segment spans "
      f"{cnt['epochs_per_segment'] / cnt['segments']:.1f} epochs on average")
for name in ("production (OCR+appearance)", "OCR only, per-epoch (as coded)", "OCR only, once per segment"):
    st, c = cnt[(name, "strong")], cnt[(name, "contradicted")]
    print(f"  {name:32s} strongly-read segments contradicted: {c:4d} / {st} ({100 * c / st:.1f} %), "
          f"{cnt[(name, 'contradicted_s')]:.0f} s;  labels differing from production: {cnt[(name, 'label_changes_vs_prod')]}")
