"""Evaluate the analysis stage (tracking, identity, attribution) on frozen cached scans.

Runs pipeline.analyse() in memory on every output/<match>/scan.pkl (never writes
to output/), in parallel, and reports:

* identity: held-out bumper-OCR agreement (identities assigned from half the reads, scored on
  the other half) - needs --holdout (3 analyses per match)
* attribution: within-alliance share correlation with fuel OPR and top-scorer agreement,
  separately for train events (casnv, joh) and test events (the rest)

Parameter overrides: --param robots.FUSE_DIST=1.6 --param attribution.PRIOR=0.1 ...
Other scans (e.g. a detector experiment's rescans): --snap data/exp_bumpers/b1
Cached bumper reads for the current segmentation (saves ~450 EasyOCR reads per analysis):
--ocr_cache data/exp_dino/ocr_cache (<match>.pkl = {crop_id: reads}, merged into md.ocr_cache)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SNAP = ROOT / "output"  # the current scans (data/eval_snapshot, a copy from 2026-10-17, was removed)
TRAIN_EVENTS = {"2026casnv", "2026joh"}

_PARAMS: list[str] = []
_OPTS: dict = {}
_OCR_CACHE: Path | None = None


def _apply(params: list[str]):
    import importlib
    for p in params:
        k, v = p.split("=", 1)
        mod, name = k.rsplit(".", 1)
        m = importlib.import_module(f"autoscout.{mod}")
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = v
        if isinstance(val, list):
            val = tuple(val)
        setattr(m, name, val)


def _init(params, opts, snap=None, ocr_cache=None):
    global _PARAMS, _OPTS, SNAP, _OCR_CACHE
    _PARAMS, _OPTS = params, opts
    if snap:
        SNAP = Path(snap)
    if ocr_cache:
        _OCR_CACHE = Path(ocr_cache)
    _apply(params)


def _load(key: str):
    with open(SNAP / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    if _OCR_CACHE is not None and (_OCR_CACHE / f"{key}.pkl").exists():
        with open(_OCR_CACHE / f"{key}.pkl", "rb") as fh:
            md.ocr_cache.update(pickle.load(fh))
    return md


def _one(args):
    key, holdout = args
    from autoscout import pipeline
    md = _load(key)
    md.cnn_cache = getattr(md, "cnn_cache", None)
    pipeline.load_dino(md, SNAP / key)
    res = pipeline.analyse(md, pipeline.Options(**_OPTS))
    info = md.info
    out = {"key": key, "event": info.event_key, "teams": {}, "agree": 0.0, "mass": 0.0}
    for al in ("red", "blue"):
        for t in info.teams(al):
            v = res["teams"][t]
            out["teams"][t] = (al, v["fuel"])
    if holdout:
        for k in (0, 1):
            md = _load(key)
            pipeline.load_dino(md, SNAP / key)
            r = pipeline.analyse(md, pipeline.Options(ocr_holdout=k, **_OPTS))
            for al in ("red", "blue"):
                teams = info.teams(al)
                for s in r["identity"][al].values():
                    if s["team"] is None:
                        continue
                    out["agree"] += s["held"][teams.index(s["team"])]
                    out["mass"] += sum(s["held"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--opt", action="append", default=[], help="Options field override, e.g. w_cls=2")
    ap.add_argument("--matches", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save")
    ap.add_argument("--snap", default=None, help="directory of <match>/scan.pkl (default output/)")
    ap.add_argument("--ocr_cache", default=None, help="directory of <match>.pkl bumper-read caches to merge")
    a = ap.parse_args()
    if a.snap:
        _init(a.param, {}, a.snap)
    from autoscout import opr

    keys = sorted(p.name for p in SNAP.iterdir() if (p / "scan.pkl").exists())
    if a.matches:
        keys = [k for k in keys if k in a.matches]
    opts = {}
    for kv in a.opt:
        k, v = kv.split("=", 1)
        try:
            opts[k] = json.loads(v)
        except json.JSONDecodeError:
            opts[k] = v
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.param, opts, a.snap, a.ocr_cache)) as ex:
        results = list(ex.map(_one, [(k, a.holdout) for k in keys]))
    oprs = {}
    for r in results:
        if r["event"] not in oprs:
            oprs[r["event"]] = opr.fuel_opr(r["event"])
    summary = {}
    for name, evs in (("train", TRAIN_EVENTS), ("test", None)):
        sa, so, top, n_al = [], [], 0, 0
        for r in results:
            if (evs is None) == (r["event"] in TRAIN_EVENTS):
                continue
            o = oprs[r["event"]]
            for al in ("red", "blue"):
                ts = [t for t, (aa, _) in r["teams"].items() if aa == al]
                f = np.array([r["teams"][t][1] for t in ts])
                op = np.array([o.get(t, np.nan) for t in ts])
                if f.sum() < 40 or not np.all(np.isfinite(op)) or op.sum() <= 0:
                    continue
                sa += list(f / f.sum())
                so += list(op / op.sum())
                top += int(np.argmax(f) == np.argmax(op))
                n_al += 1
        summary[name] = {"corr": float(np.corrcoef(sa, so)[0, 1]) if len(sa) > 5 else float("nan"),
                         "top": top / max(1, n_al), "n": n_al}
    line = (f"train corr {summary['train']['corr']:.3f} top {100 * summary['train']['top']:.0f}% | "
            f"test corr {summary['test']['corr']:.3f} top {100 * summary['test']['top']:.0f}%")
    if a.holdout:
        ag = sum(r["agree"] for r in results) / max(1e-9, sum(r["mass"] for r in results))
        summary["identity_holdout"] = ag
        line += f" | identity held-out {100 * ag:.1f}%"
    print(f"{line}   {a.param} {a.opt}", flush=True)
    if a.save:
        Path(a.save).write_text(json.dumps({"summary": summary, "params": a.param,
                                            "teams": {r["key"]: r["teams"] for r in results}}, default=str))


if __name__ == "__main__":
    main()
