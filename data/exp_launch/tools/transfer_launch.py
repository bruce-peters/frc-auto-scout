"""Does the credit model learned on Chezy scouting transfer to the other events (no scouting there)?

Fits the models on every scouted Chezy alliance, applies them to the other events' alliances, and compares
within-alliance shares against fuel-OPR shares (a noisy, team-level reference):

* video-only models (no OPR input) vs the full-event fuel OPR, the historical share-corr metric;
* models that use OPR as a prior would be scored against their own input that way, so they are scored
  split-half instead: the prior is the OPR from the event's odd-numbered quals and the reference the OPR
  from its even-numbered quals, and the other way round (both directions pooled). The OPR prior alone,
  the 0.7 video + 0.3 OPR blend and the learned model all get the same half-OPR input.

  .venv/Scripts/python data/exp_attrmodel/tools/transfer.py DS.pkl [--out JSON]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import fit_cv_launch as FC  # noqa: E402
from autoscout import opr, tba  # noqa: E402


def opr_subset(event_key: str, keep, ridge: float = 0.5) -> dict[int, float]:
    """autoscout.opr.fuel_opr restricted to the quals for which keep(match_number) is true."""
    ms = [m for m in tba.event_matches(event_key) if m.comp_level == "qm" and m.breakdown and keep(m.match_number)]
    teams = sorted({t for m in ms for t in m.red + m.blue})
    idx = {t: i for i, t in enumerate(teams)}
    rows, y = [], []
    for m in ms:
        for al in ("red", "blue"):
            hub = m.hub(al)
            if not hub:
                continue
            r = np.zeros(len(teams))
            for t in m.teams(al):
                r[idx[t]] = 1
            rows.append(r)
            y.append(hub.get("totalCount", 0))
    if not rows:
        return {}
    A, y = np.array(rows), np.array(y, dtype=float)
    mu = y.mean() / 3
    x = np.linalg.solve(A.T @ A + ridge * np.eye(len(teams)), A.T @ (y - A @ np.full(len(teams), mu)))
    x += mu
    return {t: float(max(0.0, x[idx[t]])) for t in teams}


def share_stats(pred: np.ndarray, ref: np.ndarray, tot: np.ndarray) -> dict:
    return {"corr": float(np.corrcoef(pred.ravel(), ref.ravel())[0, 1]),
            "top": float((pred.argmax(1) == ref.argmax(1)).mean()),
            "mae": float((np.abs(pred - ref) * tot[:, None]).mean()), "n": int(len(pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds")
    ap.add_argument("--out")
    ap.add_argument("--video-spec", default="lin/all+dist")
    ap.add_argument("--model-spec", default="lin/all+dist+mlprior")
    ap.add_argument("--lam", type=float, default=3e-4)
    a = ap.parse_args()
    feats, recs = FC.load(a.ds)
    ok = [r for r in recs if r.get("overlay") and len(r["n"]) > 0 and r["video_share"] is not None]
    train = [r for r in ok if "scout_n" in r and r.get("tba_total") and r["scout_n"].sum() > 0]
    sv, sm = FC.parse_spec(a.video_spec), FC.parse_spec(a.model_spec)
    th_v = FC.fit(sv, FC.Pack(train, feats), a.lam)
    th_m = FC.fit(sm, FC.Pack(train, feats), a.lam)
    print(f"fitted on {len(train)} scouted cc alliances: video-only {json.dumps({k: round(v, 2) for k, v in th_v.items()})}")
    print(f"   model {json.dumps({k: round(v, 2) for k, v in th_m.items()})}")
    out = {"theta_video": th_v, "theta_model": th_m, "events": {}}
    events = sorted({r["event"] for r in ok})
    pooled = {}
    for ev in events + ["ALL other events"]:
        er = [r for r in ok if (r["event"] == ev if ev != "ALL other events" else r["event"] != "2026cc")]
        res = {}
        if ev != "ALL other events":
            o_full = opr.fuel_opr(ev)
            halves = {"odd": opr_subset(ev, lambda n: n % 2 == 1), "even": opr_subset(ev, lambda n: n % 2 == 0)}
            pooled[ev] = (o_full, halves)
        # filter like scripts/evaluate.py: a real alliance total and an OPR for all three teams
        keep = []
        for r in er:
            o_full, halves = pooled[r["event"]]
            vals = [np.array([d.get(t, np.nan) for t in r["teams"]]) for d in (o_full, halves["odd"], halves["even"])]
            if all(np.all(np.isfinite(v)) and v.sum() > 0 for v in vals) and (r.get("tba_total") or 0) >= 40:
                keep.append((r, vals))
        if not keep:
            continue
        rs = [k[0] for k in keep]
        tot = np.array([r["tba_total"] for r in rs], float)
        ref_full = np.array([v[0] / v[0].sum() for _, v in keep])
        old = np.array([r["old_unscaled"] / r["old_unscaled"].sum() for r in rs])
        vid_learned = FC.predict(sv, th_v, FC.Pack(rs, feats))
        res["video-only, vs full OPR"] = {"old rule": share_stats(old, ref_full, tot),
                                          "learned evidence (no OPR)": share_stats(vid_learned, ref_full, tot)}
        # split-half: prior from one half, reference from the other, both directions pooled
        P_, R_, T_ = {k: [] for k in ("old rule", "OPR prior alone", "0.7 video + 0.3 OPR", "learned model")}, [], []
        for pri, rf in ((1, 2), (2, 1)):
            rs2 = []
            for r, v in keep:
                r2 = copy.copy(r)
                r2["opr"] = v[pri]
                rs2.append(r2)
            so = np.array([v[pri] / v[pri].sum() for _, v in keep])
            P_["old rule"].append(old)
            P_["OPR prior alone"].append(so)
            P_["0.7 video + 0.3 OPR"].append(0.7 * old + 0.3 * so)
            P_["learned model"].append(FC.predict(sm, th_m, FC.Pack(rs2, feats)))
            R_.append(np.array([v[rf] / v[rf].sum() for _, v in keep]))
            T_.append(tot)
        R_, T_ = np.concatenate(R_), np.concatenate(T_)
        res["split-half OPR"] = {k: share_stats(np.concatenate(v), R_, T_) for k, v in P_.items()}
        out["events"][ev] = res
        print(f"\n{ev}: {len(rs)} alliances")
        for blk, d in res.items():
            for k, s in d.items():
                print(f"   {blk:24s} {k:28s} share corr {s['corr']:.3f}  top {100 * s['top']:.0f} %  fuel MAE {s['mae']:.1f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
