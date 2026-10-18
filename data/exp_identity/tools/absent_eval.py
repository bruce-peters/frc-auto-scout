"""Run pipeline.analyse in memory (never writes output/) on output/<match>/scan.pkl for several
variants of the absent-state prototype (proto.py) or of the real flags, one scan load per match.

  absent_eval.py RUN --variants base v1 ... [--matches k1 k2 | --cc] [--holdout] [--real]

Appends one JSON line per match to data/exp_identity/runs/RUN.jsonl (resumable; keys done are skipped).
--real: variants are module-param / Options overrides of the real code (no proto hooks).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np  # noqa: E402

import proto  # noqa: E402
from autoscout import pipeline  # noqa: E402

RUNS = ROOT / "data" / "exp_identity" / "runs"
OUT = ROOT / "output"
SIDE = ROOT / "data" / "exp_identity" / "ocr_sidecar"
STUB = [0]


def _stub_read(img):
    STUB[0] += 1
    return []


# Residual injection: the production result.json holds each segment's bumper evidence computed with every
# read; evidence missing from the scan's cache is put back on that segment's uncached crops (split evenly),
# so the non-held-out analysis reproduces production identity. Held-out splits of it are approximate.
RESID: dict = {}
_ORIG_TE = None


def _team_evidence(reads, teams):
    if reads and reads[0][0] == "__RESID__":
        v = RESID.get(reads[0][1])
        return np.zeros(len(teams)) if v is None else v.copy()
    return _ORIG_TE(reads, teams)


def prepare(md, key) -> dict:
    """Stub pass -> uncached crops; spread result.json's residual evidence over them."""
    from autoscout import identity as I
    RESID.clear()
    before = set(md.ocr_cache)
    cap = {}
    orig = I.assign

    def spy(segs, evidence, switch_cost=0.6):
        if segs and segs[0].alliance not in cap:
            cap[segs[0].alliance] = (segs, {k: np.asarray(v, float).copy() for k, v in evidence.items()})
        return orig(segs, evidence, switch_cost)
    I.assign = spy
    on = proto.CFG["on"]
    proto.CFG["on"] = False
    try:
        o = pipeline.Options()
        pipeline.analyse(md, o)
    finally:
        I.assign = orig
        proto.CFG["on"] = on
    missing = set(md.ocr_cache) - before
    stats = {"missing": len(missing), "mismatch": 0, "unexplained": 0.0, "injected": 0.0}
    rp = OUT / key / "result.json"
    stored = json.loads(rp.read_text())["identity"] if rp.exists() else {}
    for al, (segs, ev) in cap.items():
        sd = stored.get(al, {})
        for s in segs:
            e = sd.get(str(s.id))
            if e is None or e["slot"] != s.slot or abs(e["t0"] - round(s.t0, 2)) > 0.011 or abs(e["t1"] - round(s.t1, 2)) > 0.011:
                stats["mismatch"] += 1
                continue
            res = np.clip(np.array(e["ocr"], float) - ev.get(s.id, np.zeros(3)) / o.w_ocr, 0, None)
            res[res < 0.006] = 0.0  # rounding of the stored values
            mc = [d.crop_id for d in s.dets if d.crop_id in missing]
            if mc and res.sum() > 0:
                for c in mc:
                    RESID[c] = res / len(mc)
                stats["injected"] += float(res.sum())
            elif res.sum() > 0.05:
                stats["unexplained"] += float(res.sum())
    for c in missing:
        md.ocr_cache[c] = [("__RESID__", c)]
    stats["unexplained"] = round(stats["unexplained"], 2)
    stats["injected"] = round(stats["injected"], 2)
    return stats

# name -> (proto CFG overrides (None = proto off), module params {"module.NAME": value}, Options overrides)
VARIANTS = {
    "base": (None, {}, {}),
    "base_p": ({"abs_cost_dead": np.inf, "zero": "none"}, {}, {}),  # proto on, nothing enabled (== base?)
    # identity only (attribution unchanged except absent zeroing switched off)
    "id_d2": ({"abs_cost_dead": 2.0, "zero": "none"}, {}, {}),
    # identity + PRIOR/HIDDEN zero in absent stretches
    "abs_d2": ({"abs_cost_dead": 2.0}, {}, {}),
    "abs_d1": ({"abs_cost_dead": 1.0}, {}, {}),
    "abs_d05": ({"abs_cost_dead": 0.5}, {}, {}),
    "abs_d4": ({"abs_cost_dead": 4.0}, {}, {}),
    "abs_d2_all4": ({"abs_cost_dead": 2.0, "abs_cost": 4.0}, {}, {}),
    "abs_d2_all8": ({"abs_cost_dead": 2.0, "abs_cost": 8.0}, {}, {}),
    "abs_d2_dh": ({"abs_cost_dead": 2.0, "dead_hidden": True}, {}, {}),
    "abs_d2_zs": ({"abs_cost_dead": 2.0, "zero_shots": True}, {}, {}),
    "abs_d2_st": ({"abs_cost_dead": 2.0, "stat_w": 20.0, "stat_d0": 1.5}, {}, {}),
    "st_only": ({"abs_cost_dead": np.inf, "zero": "none", "stat_w": 20.0, "stat_d0": 1.5}, {}, {}),
    # parked flag (median 20 s extent < park_ext m): PRIOR/HIDDEN 0 all match
    "park15": ({"abs_cost_dead": np.inf, "zero": "none", "park_ext": 1.5}, {}, {}),
    "abs_d2_park15": ({"abs_cost_dead": 2.0, "park_ext": 1.5}, {}, {}),
    # co-located concurrent segments favour a duplicate label
    "abs_d2_near": ({"abs_cost_dead": 2.0, "near_m": 1.5, "near_bonus": 2.0}, {}, {}),
    "abs_d2_near_park": ({"abs_cost_dead": 2.0, "near_m": 1.5, "near_bonus": 2.0, "park_ext": 1.5}, {}, {}),
    "abs_d2_near_dh8": ({"abs_cost_dead": 2.0, "near_m": 1.5, "near_bonus": 2.0, "dead_hidden": True,
                         "dh_min_s": 8.0}, {}, {}),
    "abs_d2_all4_near": ({"abs_cost_dead": 2.0, "abs_cost": 4.0, "near_m": 1.5, "near_bonus": 2.0}, {}, {}),
    "park15p": ({"abs_cost_dead": np.inf, "zero": "none", "park_ext": 1.5, "park_mode": "prior"}, {}, {}),
    "park10p": ({"abs_cost_dead": np.inf, "zero": "none", "park_ext": 1.0, "park_mode": "prior"}, {}, {}),
    "park25p": ({"abs_cost_dead": np.inf, "zero": "none", "park_ext": 2.5, "park_mode": "prior"}, {}, {}),
    "abs_near_dh8_park15p": ({"abs_cost_dead": 2.0, "near_m": 1.5, "near_bonus": 2.0, "dead_hidden": True,
                              "dh_min_s": 8.0, "park_ext": 1.5, "park_mode": "prior"}, {}, {}),
    "abs_near_park15p": ({"abs_cost_dead": 2.0, "near_m": 1.5, "near_bonus": 2.0, "park_ext": 1.5,
                          "park_mode": "prior"}, {}, {}),
    "abs_near_dhv50": ({"abs_cost_dead": 2.0, "near_m": 1.5, "near_bonus": 2.0, "dead_hidden": True,
                        "dh_max_vis": 0.5}, {}, {}),
    # the real implementation (pipeline.Options; run with --real)
    "R_base": (None, {}, {}),
    "R_abs": (None, {}, {"absent_cost_dead": 2.0, "absent_near_bonus": 2.0}),
    "R_abs_dh8": (None, {}, {"absent_cost_dead": 2.0, "absent_near_bonus": 2.0, "absent_dead_hidden_s": 8.0}),
    "R_park": (None, {}, {"park_ext": 1.5}),
    "R_abs_d4": (None, {}, {"absent_cost_dead": 4.0, "absent_near_bonus": 2.0}),
    "R_abs_nonear": (None, {}, {"absent_cost_dead": 2.0}),
    "R_park10": (None, {}, {"park_ext": 1.0}),
    "R_park25": (None, {}, {"park_ext": 2.5}),
    "R_abs_park": (None, {}, {"absent_cost_dead": 2.0, "absent_near_bonus": 2.0, "park_ext": 1.5}),
    "R_abs_dh8_park": (None, {}, {"absent_cost_dead": 2.0, "absent_near_bonus": 2.0, "absent_dead_hidden_s": 8.0,
                                  "park_ext": 1.5}),
    "abs_d2_dh8_park": ({"abs_cost_dead": 2.0, "dead_hidden": True, "dh_min_s": 8.0, "park_ext": 1.5}, {}, {}),
}


def free_gb() -> float:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"], capture_output=True, text=True)
    try:
        return int(out.stdout.strip()) / 1024 / 1024
    except ValueError:
        return 0.0


def wait_mem(need=6.0):
    while True:
        g = free_gb()
        if g >= need:
            return g
        print(f"   free {g:.1f} GB < {need} GB, waiting", flush=True)
        time.sleep(60)


def set_params(params: dict) -> dict:
    import importlib
    old = {}
    for k, v in params.items():
        mod, name = k.rsplit(".", 1)
        m = importlib.import_module(f"autoscout.{mod}")
        old[k] = getattr(m, name)
        setattr(m, name, v)
    return old


def run_variant(md, spec, holdout: bool, real: bool) -> dict:
    cfg, params, opts = spec
    old = set_params(params)
    saved_cfg = dict(proto.CFG)
    try:
        if cfg is None or real:
            proto.CFG["on"] = False
        else:
            proto.CFG.update(cfg)
            proto.CFG["on"] = True
        o = pipeline.Options(**opts)
        proto.reset(md)
        proto.CTX["w_ocr"] = o.w_ocr
        res = pipeline.analyse(md, o)
        rec = {"fuel": {str(t): v["fuel"] for t, v in res["teams"].items() if not str(t).startswith("_")}}
        if proto.CTX.get("parked"):
            rec["parked"] = proto.CTX["parked"]
        if "src" in proto.CTX:
            rec["src"] = {str(t): np.round(v, 1).tolist() for t, v in proto.CTX["src"].items()}
        if proto.CFG["on"]:
            rec["dead"] = {al: [int(x) for x in st.get("dead", [])] for al, st in proto.CTX["al"].items()}
            rec["absent_s"] = {al: {str(md.info.teams(al)[ti]): round(sum(b - a for a, b in iv), 1)
                                    for ti, iv in st.get("absent", {}).items()} for al, st in proto.CTX["al"].items()}
        fl = {k: res[k] for k in ("absent", "dead", "parked") if k in res}  # real implementation's flags
        if fl:
            rec["flags"] = json.loads(json.dumps(fl, default=float))
        rec["labels_s"] = {}
        for al in ("red", "blue"):
            for s in res["identity"][al].values():
                if s["team"] is not None:
                    rec["labels_s"][str(s["team"])] = round(rec["labels_s"].get(str(s["team"]), 0) + s["t1"] - s["t0"], 1)
        if holdout:
            ag = ms = 0.0
            for k in (0, 1):
                proto.reset(md)
                r = pipeline.analyse(md, pipeline.Options(ocr_holdout=k, **opts))
                for al in ("red", "blue"):
                    teams = md.info.teams(al)
                    for s in r["identity"][al].values():
                        if s["team"] is None:
                            continue
                        ag += s["held"][teams.index(s["team"])]
                        ms += sum(s["held"])
            rec["agree"], rec["mass"] = ag, ms
        return rec
    finally:
        set_params(old)
        proto.CFG.clear()
        proto.CFG.update(saved_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--variants", nargs="+", default=["base"])
    ap.add_argument("--matches", nargs="*", default=[])
    ap.add_argument("--cc", action="store_true", help="only scouted 2026cc quals")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--holdout_variants", nargs="*", default=None, help="held-out identity only for these")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--no_inject", action="store_true", help="uncached reads = no read (no residual injection)")
    ap.add_argument("--spec", default=None, help="JSON file {name: [cfg, params, opts]} adding variants")
    a = ap.parse_args()
    if a.spec:
        for k, v in json.loads(Path(a.spec).read_text()).items():
            VARIANTS[k] = tuple(v)
    if not a.real:
        proto.install()
    from autoscout import ocr
    global _ORIG_TE
    ocr.read_numbers = _stub_read
    _ORIG_TE = ocr.team_evidence
    ocr.team_evidence = _team_evidence
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{a.run}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    keys = sorted(p.parent.name for p in OUT.glob("*/scan.pkl"))
    if a.matches:
        keys = [k for k in keys if k in a.matches]
    if a.cc:
        from autoscout import scouting
        bym = scouting.by_match(scouting.load())
        keys = [k for k in keys if k.startswith("2026cc_qm") and k in bym]
    keys = [k for k in keys if k not in done]
    print(f"{len(keys)} matches to run, variants {a.variants}", flush=True)
    for i, key in enumerate(keys):
        wait_mem()
        tic = time.time()
        try:
            with open(OUT / key / "scan.pkl", "rb") as fh:
                md = pickle.load(fh)
            md.cnn_cache = getattr(md, "cnn_cache", None)
            pipeline.load_dino(md, OUT / key)
            # bumper reads that reanalyse made on the GPU but never saved: from the sidecar (ocr_sidecar.py);
            # any still missing are stubbed as "no read" (CPU EasyOCR ~2 s each), identically for every variant
            md.ocr_cache = dict(getattr(md, "ocr_cache", None) or {})
            sp = SIDE / f"{key}.pkl"
            side = pickle.load(open(sp, "rb")) if sp.exists() else {}
            md.ocr_cache.update(side)
            STUB[0] = 0
            rec = {"key": key, "event": md.info.event_key, "red": md.info.red, "blue": md.info.blue, "v": {},
                   "sidecar": len(side)}
            if not a.no_inject:
                rec["inject"] = prepare(md, key)
            for v in a.variants:
                ho = a.holdout and (a.holdout_variants is None or v in a.holdout_variants)
                rec["v"][v] = run_variant(md, VARIANTS[v], ho, a.real)
        except Exception as e:  # a scan being rewritten by another process, etc.
            print(f"{key}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        finally:
            md = None
        rec["stubbed"] = STUB[0]
        with open(path, "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(rec, default=float) + "\n")
        print(f"[{i + 1}/{len(keys)}] {key} {time.time() - tic:.0f}s", flush=True)


if __name__ == "__main__":
    main()
