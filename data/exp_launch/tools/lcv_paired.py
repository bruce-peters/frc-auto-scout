"""Each model vs the baseline at the same lambda: paired bootstrap over matches (saved LOMO preds) + 5026 gap.

  lcv_paired.py DS.pkl CV.json [BASE_SPEC]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_cv_launch as FC  # noqa: E402

feats, recs = FC.load(sys.argv[1])
d = json.loads(Path(sys.argv[2]).read_text())
BASE = sys.argv[3] if len(sys.argv) > 3 else "lin/all+dist+mlprior"
sc = [r for r in recs if r.get("overlay") and "scout_n" in r and r.get("tba_total") and r["video_share"] is not None
      and r["scout_n"].sum() > 0 and len(r["n"]) > 0]
pb = np.array(d["models"][BASE]["pred"])
mb = FC.metrics(pb, sc)
i5 = [(i, list(r["teams"]).index(5026)) for i, r in enumerate(sc) if 5026 in list(r["teams"])]
ref5 = [sc[i]["scout_n"][j] / sc[i]["scout_n"].sum() for i, j in i5]


def gap(p):
    return float(np.mean([p[i, j] - s for (i, j), s in zip(i5, ref5)]))


print(f"baseline {BASE}: TV {mb['tv']:.4f}  corr {mb['corr']:.4f}  top {mb['top']:.3f}  fuel MAE {mb['fuel_mae']:.2f}  "
      f"5026 gap {gap(pb):+.3f} (n={len(i5)})")
print(f"{'model':42s} {'coef':>18s}  {'dTV [95% CI]':>26s}  {'dcorr [95% CI]':>26s}  {'dMAE [95% CI]':>24s}  {'dtop':>6s}  5026")
for name, m in d["models"].items():
    if name == BASE:
        continue
    pv = np.array(m["pred"])
    mv = FC.metrics(pv, sc)
    ci = FC.boot_diff(pv, pb, sc)
    new = {k: v for k, v in m["theta"].items() if k.startswith(("g_", "lc_")) and k not in ("g_dist", "g_ml")}
    co = " ".join(f"{k}={v:.2f}" for k, v in new.items())
    print(f"{name:42s} {co:>18s}  {mv['tv'] - mb['tv']:+.4f} [{ci['tv'][0]:+.4f}, {ci['tv'][1]:+.4f}]  "
          f"{mv['corr'] - mb['corr']:+.4f} [{ci['corr'][0]:+.4f}, {ci['corr'][1]:+.4f}]  "
          f"{mv['fuel_mae'] - mb['fuel_mae']:+.2f} [{ci['fuel_mae'][0]:+.2f}, {ci['fuel_mae'][1]:+.2f}]  "
          f"{100 * (mv['top'] - mb['top']):+5.1f}  {gap(pv):+.3f}")
