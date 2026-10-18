"""Check that the opt-in model in autoscout/attribution.py (ATTR_MODEL) computes what was cross-validated.

A. attribution.split_increments with the model, on TeamSignals rebuilt from every stored result (as in
   build_ds.py, plus the distance samples pipeline.analyse adds), against the offline fit_cv prediction:
   all overlay alliances.
B. pipeline wiring on real scans: pipeline.analyse is run in memory (nothing is written) with the rule and
   with the model. Bumper OCR of crops missing from the scan's cache is stubbed out (CPU OCR takes tens of
   minutes per match); this changes identity for both runs alike. The model run is compared with the
   offline prediction built from the rule run's own result.

  .venv/Scripts/python -u data/exp_attrmodel/tools/parity.py MODEL.json SNAPSHOT_DIR [MATCH ...]
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

import build_ds as B  # noqa: E402
import fit_cv as FC  # noqa: E402
from autoscout import attribution as A, opr as O, pipeline as P, tba  # noqa: E402

_OPR = {}


def event_opr(ev):
    if ev not in _OPR:
        _OPR[ev] = O.fuel_opr(ev)
    return _OPR[ev]


def alliance_recs(r: dict, dets: list, info) -> list[tuple[dict, list]]:
    """(dataset record, TeamSignals with distance samples) per overlay alliance of one result."""
    tl = B.TL(r)
    dbt = defaultdict(list)
    for d in dets:
        if d[7] is None:
            continue
        al_t = "red" if d[7] in info.red else "blue"
        dbt[(d[7], "main" if d[1] == "main" else ("own" if d[1] == f"{al_t}_end" else "oth"))].append(d[0])
    dbt = {k: np.array(sorted(v)) for k, v in dbt.items()}
    out = []
    for al in ("red", "blue"):
        teams = info.teams(al)
        if not tl.counts.get(al):
            continue
        inc = A.increments_from_counts(P._match_counts(tl, al, info.hub(al) or {}))
        sigs = [B.team_signals(r, info, al, t, tl, dbt) for t in teams]
        for s in sigs:
            s.dist_t, s.dist_v = s.grid, s.dist
        X, N, PER = [], [], []
        for t, n in inc:
            if n <= 0:
                continue
            mt = tl.match_time(t - 2.0)
            per = A.period_of(mt) if mt is not None else "unknown"
            X.append(np.stack([B.robot_feats(s, t, A.AUTO_ZONE_GATE or per != "auto") for s in sigs]))
            N.append(n)
            PER.append(per)
        o = event_opr(info.event_key)
        rec = {"key": info.key, "al": al, "teams": teams, "X": np.array(X), "n": np.array(N, float),
               "period": np.array(PER, dtype=object), "opr": np.array([o.get(t, np.nan) for t in teams]),
               "typ": np.full(3, np.nan), "inc": inc, "tl": tl}
        out.append((rec, sigs))
    return out


def model_shares_impl(rec, sigs, model_path):
    old = A.ATTR_MODEL
    try:
        A.ATTR_MODEL = model_path
        sp = A.split_increments(rec["inc"], sigs, rec["tl"].match_time, opr=event_opr(tba.match(rec["key"]).event_key))
    finally:
        A.ATTR_MODEL = old
    f = np.array([sum(sp[t].values()) for t in rec["teams"]])
    return f / f.sum()


def shares(res, teams):
    f = np.array([res["teams"][t]["fuel"] for t in teams], float)
    return f / f.sum() if f.sum() > 0 else f


def main():
    model_path, snap, keys = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
    m = json.loads(Path(model_path).read_text())
    spec, th = FC.parse_spec(m["fit"]["spec"]), m["fit"]["theta"]
    feats = B.FEATS
    # ---- A: attribution implementation vs offline model, every stored overlay alliance
    diffs = []
    for p in sorted(snap.glob("2026*/result.json")):
        r = json.loads(p.read_text())
        info = tba.match(r["match"])
        dets = json.loads((p.parent / "dets.json").read_text())
        for rec, sigs in alliance_recs(r, dets, info):
            if not len(rec["n"]):
                continue
            off = FC.predict(spec, th, FC.Pack([rec], feats))[0]
            imp = model_shares_impl(rec, sigs, model_path)
            diffs.append(float(np.abs(off - imp).max()))
    diffs = np.array(diffs)
    print(f"A. split_increments(model) vs offline model on {len(diffs)} alliances: max |share diff| "
          f"{diffs.max():.2e}, mean {diffs.mean():.2e}", flush=True)
    # ---- B: pipeline wiring
    P.ocr.read_numbers = lambda img: []  # OCR only what the scan already cached (see docstring)
    worst = 0.0
    for k in keys:
        tic = time.time()
        out_dir = ROOT / "output" / k
        with open(out_dir / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        P.load_dino(md, out_dir)
        info = md.info
        old = A.ATTR_MODEL
        try:
            A.ATTR_MODEL = None
            r0 = P.analyse(md, P.Options())
            A.ATTR_MODEL = model_path
            r1 = P.analyse(md, P.Options())
        finally:
            A.ATTR_MODEL = old
        r0j = json.loads(json.dumps(r0, default=P._json_default))
        dets0 = json.loads(json.dumps(md.det_rows, default=P._json_default))
        for rec, sigs in alliance_recs(r0j, dets0, info):
            teams = rec["teams"]
            off = FC.predict(spec, th, FC.Pack([rec], feats))[0]
            s0, s1 = shares(r0, teams), shares(r1, teams)
            d = float(np.abs(s1 - off).max())
            worst = max(worst, d)
            print(f"B. {k} {rec['al']}: rule {s0.round(3)}  model via pipeline {s1.round(3)}  model offline "
                  f"{off.round(3)}  max diff {d:.3f};  alliance fuel rule "
                  f"{sum(r0['teams'][t]['fuel'] for t in teams):.1f} / model {sum(r1['teams'][t]['fuel'] for t in teams):.1f}",
                  flush=True)
        print(f"   {k}: {time.time() - tic:.0f} s, credit_model in result: {r1.get('credit_model')}", flush=True)
        del md
    if keys:
        print(f"B. largest share difference pipeline vs offline: {worst:.3f}")


if __name__ == "__main__":
    main()
