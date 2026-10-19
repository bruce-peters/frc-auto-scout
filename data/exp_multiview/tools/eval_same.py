"""Same-alliance / cross-alliance split of the hard two-view pick on the c24 and fresh sets (eval_mv rules)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import eval_mv as EM  # noqa: E402
import eval_truth as E  # noqa: E402

if len(sys.argv) > 1:
    C.PAIRS_DIR = sys.argv[1]
truth, matched = E.c24_cases()
out = {}
for name, items in (EM.c24_like(matched, "c24"), EM.c24_like(E.fresh_cases(), "fresh")):
    cr = np.array([it["cross"] for it in items])
    print(f"\n{name}: same-alliance n={(~cr).sum()}, cross n={cr.sum()}")
    for rn in ("rms<=6", "rms<=10", "norm<=2 (main 8, end 4 px)", "any, mutual strict"):
        keys, ot, on, cov, picks = EM.evaluate(name, items, EM.RULES[rn], EM.GATES["gate"])
        s = ~cr
        lo, hi = E.boot_diff(list(keys[s]), on[s], ot[s])
        print(f"   {rn:28s} same-alliance: cover {100 * cov[s].mean():5.1f}%  two-view {100 * on[s & cov].mean():5.1f} vs today {100 * ot[s & cov].mean():5.1f} on covered;"
              f"  all same {100 * ot[s].mean():5.1f} -> {100 * on[s].mean():5.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}];  cross {100 * ot[cr].mean():5.1f} -> {100 * on[cr].mean():5.1f}")
        out[f"{name} | {rn}"] = {"same_cover": float(cov[s].mean()), "same_mv": float(on[s & cov].mean()), "same_today_cov": float(ot[s & cov].mean()),
                                 "same_today": float(ot[s].mean()), "same_new": float(on[s].mean()), "same_ci": [float(lo), float(hi)],
                                 "cross_today": float(ot[cr].mean()), "cross_new": float(on[cr].mean())}
json.dump(out, open(C.EXP / f"eval_same_{C.PAIRS_DIR}.json", "w"), indent=1)
