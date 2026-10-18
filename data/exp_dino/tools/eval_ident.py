"""Fast held-out identity (+ optional scouting / fuel-OPR) evaluation of identity-evidence configs.

Same metric as scripts/eval_analysis.py --holdout (identities assigned from the bumper reads of crops with
id % 2 != k, scored on the reads of the other half, k = 0, 1), but much faster on one process:
* each scan is unpickled once and every config is run on it;
* the bumper reads made for the current segmentation are cached in data/exp_dino/ocr_cache/<match>.pkl
  (eval_analysis.py re-reads ~450 crops per analysis because the cache in scan.pkl predates recalibration);
* the held-out runs stop after the identity stage (only res["identity"] is scored).
With --full, one complete analysis per config also gives the fuel split: share correlation vs fuel OPR
(eval_analysis.py's train/test split) and, on the scouted Chezy quals, scripts/score_scouting.score().
Never writes output/.

  eval_ident.py --cfg base: --cfg d4:w_dino=4,probe=box_fg_ocr [--cfg gal:gal=1] [--full] --tag NAME
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402

from autoscout import fuel, opr, pipeline, scouting  # noqa: E402

EXP = ROOT / "data" / "exp_dino"
OCRC, LAB = EXP / "ocr_cache", EXP / "labels"
TRAIN_EVENTS = {"2026casnv", "2026joh"}
WRITE_ALL = False


class _Stop(Exception):
    pass


def _stop(*a, **k):
    raise _Stop


def identity_only(md, opts) -> dict:
    """pipeline.analyse up to and including identity assignment; returns res['identity']."""
    orig = fuel.is_shot
    fuel.is_shot = _stop
    try:
        return pipeline.analyse(md, opts)["identity"]
    except _Stop:
        tb = sys.exc_info()[2]
        while tb is not None:
            if tb.tb_frame.f_code.co_name == "analyse":
                return tb.tb_frame.f_locals["ident_debug"]
            tb = tb.tb_next
        raise
    finally:
        fuel.is_shot = orig


def parse_cfg(s: str) -> tuple[str, dict]:
    name, _, rest = s.partition(":")
    kv = {}
    for part in filter(None, rest.split(",")):
        k, v = part.split("=", 1)
        try:
            kv[k] = json.loads(v)
        except json.JSONDecodeError:
            kv[k] = v
    return name, kv


_GAL: dict = {}


def gallery_contrib(event: str) -> dict:
    """build_galleries.py rule on the current scans: {team: {match: (sum vec, weight)}} for one event."""
    if event in _GAL:
        return _GAL[event]
    out: dict = {}
    for p in sorted(LAB.glob(f"{event}_*.npz")):
        L = np.load(p)
        teams6 = [int(t) for t in L["teams"]]
        seg_info = {}
        for s, a, t, o in zip(L["seg"], L["al"], L["team"], L["ocr"]):
            if s >= 0 and a >= 0:
                seg_info[int(s)] = (int(a), int(t), o)
        for sid, vec, n in zip(L["seg_ids"], L["seg_res"], L["seg_n"]):
            if n == 0 or int(sid) not in seg_info:
                continue
            a, t, o = seg_info[int(sid)]
            tri = teams6[3 * a:3 * a + 3]
            if t not in tri or o.sum() < 2 or o[tri.index(t)] < 0.8 * o.sum():
                continue
            v = vec.astype(np.float64)
            v /= np.linalg.norm(v) + 1e-9
            w = float(o.sum())
            d = out.setdefault(t, {})
            sv, sw = d.get(p.stem, (np.zeros_like(v), 0.0))
            d[p.stem] = (sv + w * v, sw + w)
    _GAL[event] = out
    return out


def galleries_for(key: str, event: str) -> dict:
    out = {}
    for team, d in gallery_contrib(event).items():
        sv, sw = None, 0.0
        for m, (v, w) in d.items():
            if m == key:
                continue
            sv = v.copy() if sv is None else sv + v
            sw += w
        if sv is not None and sw > 0:
            out[team] = sv / (np.linalg.norm(sv) + 1e-9)
    return out


def opts_for(cfg: dict, key: str, event: str) -> dict:
    o = {k: v for k, v in cfg.items() if k not in ("probe", "gal")}
    pipeline.DINO_PROBE_DIR = EXP / "probe" / str(cfg.get("probe", "default"))
    if cfg.get("gal"):
        o["galleries"] = galleries_for(key, event)
    return o


def coverage(ident: dict, teams_by_al: dict) -> dict:
    """Per team: share of the alliance's tracked time span covered by segments labelled with it."""
    out = {}
    for al, teams in teams_by_al.items():
        segs = list(ident[al].values())
        if not segs:
            continue
        t0, t1 = min(s["t0"] for s in segs), max(s["t1"] for s in segs)
        grid = np.arange(t0, t1, 0.5)
        for t in teams:
            on = np.zeros(len(grid), bool)
            for s in segs:
                if s["team"] == t:
                    on |= (grid >= s["t0"]) & (grid <= s["t1"])
            out[int(t)] = float(on.mean()) if len(grid) else 0.0
    return out


def run(keys: list[str], cfgs: list[tuple[str, dict]], full: bool, tag: str, holdout: bool = True) -> dict:
    scouted = set(scouting.by_match(scouting.load())) if full else set()
    out = {name: {} for name, _ in cfgs}
    OCRC.mkdir(parents=True, exist_ok=True)
    for i, key in enumerate(keys):
        tic = time.time()
        with open(ROOT / "output" / key / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        md.cnn_cache = getattr(md, "cnn_cache", None)
        cp = OCRC / f"{key}.pkl"
        if cp.exists():
            with open(cp, "rb") as fh:
                md.ocr_cache.update(pickle.load(fh))
        n0 = len(md.ocr_cache)
        info = md.info
        for name, cfg in cfgs:
            r = {"event": info.event_key, "agree": 0.0, "mass": 0.0}
            if full:
                res = pipeline.analyse(md, pipeline.Options(**opts_for(cfg, key, info.event_key)))
                r["teams"] = {int(t): (al, res["teams"][t]["fuel"]) for al in ("red", "blue") for t in info.teams(al)}
                r["cov"] = coverage(res["identity"], {al: info.teams(al) for al in ("red", "blue")})
                if key in scouted or WRITE_ALL:
                    od = EXP / "res" / f"{tag}_{name}" / key
                    od.mkdir(parents=True, exist_ok=True)
                    (od / "result.json").write_text(json.dumps(res, default=pipeline._json_default))
                    if WRITE_ALL:  # + dets.json, for data/exp_attrmodel/tools/build_ds.py
                        (od / "dets.json").write_text(json.dumps(md.det_rows, default=pipeline._json_default))
            for k in ((0, 1) if holdout else ()):
                ident = identity_only(md, pipeline.Options(ocr_holdout=k, **opts_for(cfg, key, info.event_key)))
                for al in ("red", "blue"):
                    teams = info.teams(al)
                    for s in ident[al].values():
                        if s["team"] is None:
                            continue
                        r["agree"] += s["held"][teams.index(s["team"])]
                        r["mass"] += sum(s["held"])
            out[name][key] = r
        if len(md.ocr_cache) > n0:
            new = {c: v for c, v in md.ocr_cache.items()}
            with open(cp, "wb") as fh:
                pickle.dump(new, fh)
        line = " ".join(f"{n}={100 * out[n][key]['agree'] / max(out[n][key]['mass'], 1e-9):.1f}" for n, _ in cfgs)
        print(f"{time.strftime('%H:%M')} [{i + 1}/{len(keys)}] {key:20s} {line} ({time.time() - tic:.0f}s)", flush=True)
        del md
    return out


def opr_corr(res: dict) -> dict:
    oprs = {}
    summ = {}
    for nm, evs in (("train", TRAIN_EVENTS), ("test", None)):
        sa, so = [], []
        for key, r in res.items():
            if "teams" not in r or (evs is None) == (r["event"] in TRAIN_EVENTS):
                continue
            if r["event"] not in oprs:
                oprs[r["event"]] = opr.fuel_opr(r["event"])
            o = oprs[r["event"]]
            for al in ("red", "blue"):
                ts = [t for t, (aa, _) in r["teams"].items() if aa == al]
                f = np.array([r["teams"][t][1] for t in ts])
                op = np.array([o.get(t, np.nan) for t in ts])
                if f.sum() < 40 or not np.all(np.isfinite(op)) or op.sum() <= 0:
                    continue
                sa += list(f / f.sum())
                so += list(op / op.sum())
        summ[nm] = float(np.corrcoef(sa, so)[0, 1]) if len(sa) > 5 else float("nan")
    return summ


def summarize(out: dict, full: bool, tag: str) -> dict:
    import score_scouting
    summ = {}
    for name, res in out.items():
        s = {"identity": sum(r["agree"] for r in res.values()) / max(1e-9, sum(r["mass"] for r in res.values())),
             "matches": len(res), "by_event": {}}
        for ev in sorted({r["event"] for r in res.values()}):
            rr = [r for r in res.values() if r["event"] == ev]
            s["by_event"][ev] = sum(r["agree"] for r in rr) / max(1e-9, sum(r["mass"] for r in rr))
        if full:
            s["opr_corr"] = opr_corr(res)
            cv = [c for r in res.values() for c in r.get("cov", {}).values()]
            s["team_cov"] = {"mean": float(np.mean(cv)) if cv else None,
                             "below_50": float(np.mean(np.array(cv) < 0.5)) if cv else None}
            d = EXP / "res" / f"{tag}_{name}"
            if d.exists():
                sc = score_scouting.score(d)
                if sc:
                    s["scouting"] = {"team_matches": sc["team_matches"], "mae": sc["video"]["mae"],
                                     "corr": sc["video"]["corr"], "median_rel": sc["video"]["median_rel"],
                                     "top": sc["top_scorer"]["video"],
                                     "burst": sc.get("burst_overlap", {}).get("video")}
        summ[name] = s
        ev_s = " ".join(f"{e[4:]} {100 * v:.1f}" for e, v in s["by_event"].items())
        line = f"{name:10s} identity {100 * s['identity']:.2f}%  [{ev_s}]"
        if full:
            line += f"  opr corr train {s['opr_corr']['train']:.3f} test {s['opr_corr']['test']:.3f}"
            if s["team_cov"]["mean"] is not None:
                line += f"  team cov {100 * s['team_cov']['mean']:.1f}% (<50%: {100 * s['team_cov']['below_50']:.1f}%)"
            if "scouting" in s:
                sc = s["scouting"]
                line += (f"  scouting n={sc['team_matches']} mae {sc['mae']:.1f} corr {sc['corr']:.3f} "
                         f"top {100 * sc['top']:.0f}% burst {100 * (sc['burst'] or 0):.0f}%")
        print(line, flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", action="append", required=True, help="name:opt=val,... (probe=DIR, gal=1)")
    ap.add_argument("--matches", nargs="*", default=[])
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--no_holdout", action="store_true")
    ap.add_argument("--write_all", action="store_true", help="write result.json + dets.json for every match")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    keys = sorted(p.parent.name for p in (ROOT / "output").glob("*/scan.pkl"))
    if a.matches:
        keys = [k for k in keys if k in a.matches]
    if a.events:
        keys = [k for k in keys if k.split("_")[0] in a.events]
    cfgs = [parse_cfg(c) for c in a.cfg]
    global WRITE_ALL
    WRITE_ALL = a.write_all
    out = run(keys, cfgs, a.full, a.tag, not a.no_holdout)
    summ = summarize(out, a.full, a.tag)
    (EXP / "eval").mkdir(parents=True, exist_ok=True)
    (EXP / "eval" / f"{a.tag}.json").write_text(json.dumps({"cfgs": dict(cfgs), "summary": summ, "per_match": out},
                                                           default=str, indent=1))


if __name__ == "__main__":
    main()
