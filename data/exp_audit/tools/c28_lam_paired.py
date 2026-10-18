"""Cycle 28: volexp vs baseline at the same lambda, paired bootstrap over matches (saved LOMO preds).

  c28_lam_paired.py DS.pkl CV.json [CV.json ...]   (each from fit_cv_vol.py --lam L --specs "base;base+volexp")
"""
import json
import sys
from pathlib import Path

import numpy as np

import fit_cv_vol2 as FC  # = fit_cv.py at 09:44 (absent state) + volume extras; pres/hid_on default to 1 on older datasets

BASE, VOL = "lin/all+dist+mlprior", "lin/all+dist+mlprior+volexp"
feats, recs = FC.load(sys.argv[1])
sc = [r for r in recs if r.get("overlay") and "scout_n" in r and r.get("tba_total") and r["video_share"] is not None
      and r["scout_n"].sum() > 0 and len(r["n"]) > 0]
for f in sys.argv[2:]:
    d = json.loads(Path(f).read_text())
    pb, pv = np.array(d["models"][BASE]["pred"]), np.array(d["models"][VOL]["pred"])
    assert len(pb) == len(sc)
    mb, mv = FC.metrics(pb, sc), FC.metrics(pv, sc)
    ci = FC.boot_diff(pv, pb, sc)
    lam = d["models"][VOL]["lams"]
    print(f"{Path(f).stem}: lambda {sorted(set(lam))}  g_vol {d['models'][VOL]['theta']['g_vol']:.3f}")
    for k in ("tv", "corr", "top", "fuel_mae"):
        print(f"   {k:8s} base {mb[k]:.4f}  vol {mv[k]:.4f}  diff {mv[k] - mb[k]:+.4f}  95% CI [{ci[k][0]:+.4f}, {ci[k][1]:+.4f}]")
    # 5026 share gap vs scouting (predicted - scouted share, as c20_verify / c26_v4)
    i5 = [(i, list(r["teams"]).index(5026)) for i, r in enumerate(sc) if 5026 in list(r["teams"])]
    ref5 = [sc[i]["scout_n"][j] / sc[i]["scout_n"].sum() for i, j in i5]
    gb = np.mean([pb[i, j] - s for (i, j), s in zip(i5, ref5)]); gv = np.mean([pv[i, j] - s for (i, j), s in zip(i5, ref5)])
    print(f"   5026 share gap vs scouting: base {gb:+.3f}  vol {gv:+.3f}  (n={len(i5)})")
