"""Score every arm of a link_x_ab.py run against 5026's scouting, paired against the "base" arm.

Per arm: team-match fuel MAE / median relative error / corr / top scorer (scripts/score_scouting.score, with the
shipped credit model, not refit), the paired change in fuel MAE vs base (bootstrap over matches), and how much
fuel went to robots with no linked launch in the match.

  .venv/Scripts/python data/exp_audit/tools/score_arms.py data/exp_elim
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from score_scouting import score  # noqa: E402

dst = Path(sys.argv[1])
arms = sorted((p for p in dst.iterdir() if p.is_dir()), key=lambda p: (p.name != "base", p.name))


def zero_launch_share(d: Path) -> float:
    tot = zero = 0.0
    for rp in d.glob("*/result.json"):
        r = json.loads(rp.read_text())
        nl = Counter(s["team"] for s in r.get("shots", []) if s["team"] is not None)
        for tm, v in r["teams"].items():
            if str(tm).isdigit():
                tot += v["fuel"]
                zero += v["fuel"] if nl.get(int(tm), 0) == 0 else 0.0
    return zero / max(tot, 1e-9)


base = None
for d in arms:
    s = score(d, keep_rows=True)
    rows = s["rows"]
    err = {(r["key"], r["team"]): abs(r["video"] - r["ref"]) for r in rows}
    line = (f"{d.name:<80s} MAE {s['video']['mae']:5.1f}  med {100 * s['video']['median_rel']:3.0f} %  "
            f"corr {s['video']['corr']:.3f}  top {100 * s['top_scorer']['video']:3.0f} %  "
            f"zero-launch fuel {100 * zero_launch_share(d):4.1f} %")
    if base is None:
        base = err
    else:
        ks = sorted(set(base) & set(err))
        mk = np.array([k[0] for k in ks])
        dd = np.array([err[k] - base[k] for k in ks])
        uk = sorted(set(mk))
        per = np.array([(dd[mk == k].sum(), (mk == k).sum()) for k in uk])
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(4000):
            j = rng.integers(0, len(uk), len(uk))
            bs.append(per[j, 0].sum() / per[j, 1].sum())
        line += (f"  vs base {dd.mean():+.2f} [{np.percentile(bs, 2.5):+.2f} .. {np.percentile(bs, 97.5):+.2f}] "
                 f"better/worse matches {int((per[:, 0] < 0).sum())}/{int((per[:, 0] > 0).sum())}")
    print(line, flush=True)
