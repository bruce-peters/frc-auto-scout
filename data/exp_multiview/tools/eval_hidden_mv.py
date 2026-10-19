"""Hidden-robot subsets of the hidden truth set: does the two-view pick credit a teammate that has no box in the
view (e.g. 254 behind 3847 in the blue end camera)? Subsets as eval_hidden.py; rules from eval_mv.RULES."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_mv as EM  # noqa: E402
import eval_truth as E  # noqa: E402
import common as C  # noqa: E402

name, items = EM.hidden_set()
subsets = {
    "all": lambda it: True,
    "teammate behind the winner": lambda it: bool(it["beh"]),
    "... end cameras": lambda it: bool(it["beh"]) and it["view"] != "main",
    "... main camera": lambda it: bool(it["beh"]) and it["view"] == "main",
    "truth is a box-less teammate": lambda it: it["truth"] in it["hid"],
}
rules = ["any fit", "rms<=10", "rms<=6", "norm<=2 (main 8, end 4 px)", "norm<=1.5, ratio 1.5", "any, mutual strict"]
out = {}
for sn, sel in subsets.items():
    its = [it for it in items if sel(it)]
    print(f"\n{sn}: n={len(its)} in {len({it['key'] for it in its})} matches; truth box-less in "
          f"{100 * np.mean([it['truth'] in it['hid'] for it in its]):.0f} %")
    for rn in rules:
        for gn, gate in EM.GATES.items():
            for bo in (False, True):
                keys, ot, on, cov, picks = EM.evaluate(name, its, EM.RULES[rn], gate, boxes_only=bo)
                if cov.sum() == 0:
                    continue
                n_hid = sum(1 for p, it in zip(picks, its) if p is not None and p in it["hid"])
                ok_hid = sum(1 for p, it in zip(picks, its) if p is not None and p in it["hid"] and p == it["truth"])
                lo, hi = E.boot_diff(list(keys), on, ot)
                tag = f"{rn} | {gn}{' | boxes only' if bo else ''}"
                print(f"   {tag:52s} cover {100 * cov.mean():5.1f}%  mv {100 * on[cov].mean():5.1f}% vs today {100 * ot[cov].mean():5.1f}%"
                      f"  | all {100 * ot.mean():5.1f} -> {100 * on.mean():5.1f} {100 * (on.mean() - ot.mean()):+5.1f} [{100 * lo:+5.1f}, {100 * hi:+5.1f}]"
                      f"  box-less picks {n_hid} ({ok_hid} right)")
                out[f"{sn} | {tag}"] = {"n": len(its), "cover": float(cov.mean()), "mv": float(on[cov].mean()),
                                        "today_cover": float(ot[cov].mean()), "today": float(ot.mean()), "new": float(on.mean()),
                                        "ci": [float(lo), float(hi)], "hidden_picks": n_hid, "hidden_right": ok_hid}
json.dump(out, open(C.EXP / "eval_hidden_mv.json", "w"), indent=1)
