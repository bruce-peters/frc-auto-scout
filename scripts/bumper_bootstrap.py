"""Paired bootstrap (over matches) of the bumper experiment's identity and attribution metrics.

Reruns scripts/eval_analysis.py's per-match evaluation (_one, with the OCR holdout) for each
configuration of scripts/bumper_experiment.py, then resamples the 14 matches with replacement
(the same resample for every configuration) to give 90 % intervals for each metric and for the
difference against the whole-robot v1 baseline. Writes data/exp_bumpers/bootstrap.json.

Usage: python scripts/bumper_bootstrap.py [--workers 3] [--n 2000]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import eval_analysis  # noqa: E402

EXP = ROOT / "data" / "exp_bumpers"
CONFIGS = {  # name -> (scan dir, --param list)
    "robot": (ROOT / "data" / "eval_snapshot", []),
    "robot_v2": (EXP / "robot_v2", []),
    "bumper": (EXP / "b1", []),
    "bumper_push0": (EXP / "b1", ["pipeline.ROBOT_HALF_DEPTH=0.0"]),
}


def per_match(name: str, keys: list[str], workers: int) -> dict:
    snap, params = CONFIGS[name]
    with ProcessPoolExecutor(workers, initializer=eval_analysis._init, initargs=(params, {}, str(snap))) as ex:
        return {r["key"]: r for r in ex.map(eval_analysis._one, [(k, True) for k in keys])}


def metrics(rows: list[dict], oprs: dict) -> tuple[float, float, float]:
    """identity held-out agreement, share correlation vs OPR, top-scorer agreement."""
    agree = sum(r["agree"] for r in rows)
    mass = sum(r["mass"] for r in rows)
    sa, so, top = [], [], []
    for r in rows:
        o = oprs[r["event"]]
        for al in ("red", "blue"):
            ts = [t for t, (aa, _) in r["teams"].items() if aa == al]
            f = np.array([r["teams"][t][1] for t in ts])
            op = np.array([o.get(int(t), o.get(t, np.nan)) for t in ts], dtype=float)
            if f.sum() < 40 or not np.all(np.isfinite(op)) or op.sum() <= 0:
                continue
            sa += list(f / f.sum())
            so += list(op / op.sum())
            top.append(float(np.argmax(f) == np.argmax(op)))
    corr = float(np.corrcoef(sa, so)[0, 1]) if len(sa) > 5 else float("nan")
    return agree / max(mass, 1e-9), corr, float(np.mean(top)) if top else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    from autoscout import opr

    keys = sorted(p.name for p in (EXP / "b1").iterdir() if (p / "scan.pkl").exists())
    res = {name: per_match(name, keys, a.workers) for name in CONFIGS}
    oprs = {}
    for r in res["robot"].values():
        if r["event"] not in oprs:
            oprs[r["event"]] = opr.fuel_opr(r["event"])
    rng = np.random.default_rng(0)
    samples = [rng.choice(len(keys), len(keys), replace=True) for _ in range(a.n)]
    out = {"matches": keys, "configs": {}}
    base = None
    for name in CONFIGS:
        rows = [res[name][k] for k in keys]
        point = metrics(rows, oprs)
        boot = np.array([metrics([rows[i] for i in s], oprs) for s in samples])
        if base is None:
            base = boot
        d = boot - base
        out["configs"][name] = {
            "point": dict(zip(("identity", "corr", "top"), point)),
            "ci90": {m: np.nanpercentile(boot[:, j], [5, 95]).round(3).tolist() for j, m in enumerate(("identity", "corr", "top"))},
            "diff_vs_robot_ci90": {m: np.nanpercentile(d[:, j], [5, 95]).round(3).tolist() for j, m in enumerate(("identity", "corr", "top"))},
            "p_better_than_robot": {m: float(np.nanmean(d[:, j] > 0)) for j, m in enumerate(("identity", "corr", "top"))},
            "ocr_mass_per_match": float(np.mean([r["mass"] for r in rows])),
        }
        c = out["configs"][name]
        print(name, {k: round(v, 3) for k, v in c["point"].items()}, "diff CI", c["diff_vs_robot_ci90"],
              "P(better)", {k: round(v, 2) for k, v in c["p_better_than_robot"].items()},
              "OCR mass/match", round(c["ocr_mass_per_match"], 1), flush=True)
    (EXP / "bootstrap.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
