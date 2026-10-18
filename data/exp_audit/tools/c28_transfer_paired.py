"""Cycle 28 step 2: paired base-vs-volume comparison on the other events (no refit).

Rescores both arms of transfer_vol.py from the thetas saved in transfer_base.json / transfer_vol.json, per
alliance, with the same filter and the same OPR references, then bootstraps the differences (vol - base) over
matches of the four non-Chezy events. First prints the pooled point values to check they reproduce the JSONs.

  c28_transfer_paired.py DS.pkl BASE.json VOL.json
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

import fit_cv_vol as FC
import transfer_vol as TV
from autoscout import opr

ds, jb, jv = sys.argv[1], json.loads(Path(sys.argv[2]).read_text()), json.loads(Path(sys.argv[3]).read_text())
ARMS = {"base": (jb, "lin/all+dist", "lin/all+dist+mlprior"), "vol": (jv, "lin/all+dist+volexp", "lin/all+dist+mlprior+volexp")}
feats, recs = FC.load(ds)
ok = [r for r in recs if r.get("overlay") and len(r["n"]) > 0 and r["video_share"] is not None and r["event"] != "2026cc"]
keep = []
for ev in sorted({r["event"] for r in ok}):
    o_full = opr.fuel_opr(ev)
    odd, even = TV.opr_subset(ev, lambda n: n % 2 == 1), TV.opr_subset(ev, lambda n: n % 2 == 0)
    for r in ok:
        if r["event"] != ev:
            continue
        vals = [np.array([d.get(t, np.nan) for t in r["teams"]]) for d in (o_full, odd, even)]
        if all(np.all(np.isfinite(v)) and v.sum() > 0 for v in vals) and (r.get("tba_total") or 0) >= 40:
            keep.append((r, vals))
rs = [k[0] for k in keep]
tot = np.array([r["tba_total"] for r in rs], float)
ref_full = np.array([v[0] / v[0].sum() for _, v in keep])
mkeys = sorted({r["key"] for r in rs})
mi = np.array([mkeys.index(r["key"]) for r in rs])

pred = {}
for arm, (j, vs, ms) in ARMS.items():
    sv, sm = FC.parse_spec(vs), FC.parse_spec(ms)
    vid = FC.predict(sv, j["theta_video"], FC.Pack(rs, feats))
    halves, refs = [], []
    for pri, rf in ((1, 2), (2, 1)):
        rs2 = []
        for r, v in keep:
            r2 = copy.copy(r); r2["opr"] = v[pri]; rs2.append(r2)
        halves.append(FC.predict(sm, j["theta_model"], FC.Pack(rs2, feats)))
        refs.append(np.array([v[rf] / v[rf].sum() for _, v in keep]))
    pred[arm] = {"video": vid, "model": np.concatenate(halves)}
ref_half = np.concatenate(refs); tot2 = np.concatenate([tot, tot]); mi2 = np.concatenate([mi, mi])
BLK = {"video": ("video-only, vs full OPR", "learned evidence (no OPR)", ref_full, tot, mi),
       "model": ("split-half OPR", "learned model", ref_half, tot2, mi2)}

print(f"{len(rs)} alliances, {len(mkeys)} matches (cagle/cancmp/casnv/joh)")
for arm, (j, _, _) in ARMS.items():
    for b, (blk, name, ref, tt, _) in BLK.items():
        s = TV.share_stats(pred[arm][b], ref, tt); js = j["events"]["ALL other events"][blk][name]
        print(f"  {arm:4s} {b:5s} corr {s['corr']:.4f} (json {js['corr']:.4f})  top {s['top']:.3f} ({js['top']:.3f})  "
              f"MAE {s['mae']:.2f} ({js['mae']:.2f})")

rng = np.random.default_rng(1)
B = 4000
for b, (blk, name, ref, tt, m) in BLK.items():
    rows_of = [np.where(m == k)[0] for k in range(len(mkeys))]
    d = {"corr": [], "top": [], "mae": []}
    for _ in range(B):
        ix = np.concatenate([rows_of[k] for k in rng.integers(0, len(mkeys), len(mkeys))])
        sb = TV.share_stats(pred["base"][b][ix], ref[ix], tt[ix]); sv_ = TV.share_stats(pred["vol"][b][ix], ref[ix], tt[ix])
        for k in d:
            d[k].append(sv_[k] - sb[k])
    sb = TV.share_stats(pred["base"][b], ref, tt); sv_ = TV.share_stats(pred["vol"][b], ref, tt)
    print(f"\n{blk} / {name}: vol - base, paired bootstrap over {len(mkeys)} matches, 95 % CI")
    for k, v in d.items():
        print(f"  {k:4s} {sv_[k] - sb[k]:+.4f}  [{np.percentile(v, 2.5):+.4f}, {np.percentile(v, 97.5):+.4f}]  "
              f"P(vol worse) {np.mean(np.array(v) > 0) if k == 'mae' else np.mean(np.array(v) < 0):.2f}")
    # how far the two arms' shares move at all
    dd = np.abs(pred["vol"][b] - pred["base"][b])
    print(f"  mean |share change| {dd.mean():.4f}, alliances with a changed top robot "
          f"{np.mean(pred['vol'][b].argmax(1) != pred['base'][b].argmax(1)):.1%}")
