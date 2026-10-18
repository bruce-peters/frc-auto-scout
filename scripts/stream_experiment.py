"""Separate balls inside streams / volleys: tracker and blob-source variants, scored against scouting.

Failure-finder cycle 17: in volleys, launch arcs per scouted ball fall 4x (0.31 -> 0.08 credited per
ball). A look at 5026's volleys shows the balls are mostly separate blobs; the arcs are lost in the
flight tracker: tracks hop from one ball to the next (non-smooth fits), and a new one-point track
(zero-velocity prediction) grabs the following ball, which has just moved into almost the same spot.

Variants are re-tracked offline from the blob logs of scripts/fuel_det_experiment.py
(data/exp_fueldet/blobs/<key>.pkl: every centre frame's motion blobs + flying-fuel detections), put
into the current scan (output/<key>/scan.pkl, read only) and analysed with the current
pipeline.analyse(). Scored on:
  * credited arcs per scouted ball by burst size (failure-finder c17_stages.py bursts: time_valid scout
    rows, tap - 0.8 s, gaps > 1.5 s split a burst; arcs credited to that team within +-1 s)
  * scout share corr / MAE (cc quals with all three robots scouted) and the 5026 gap
  * fuel-OPR share corr (scripts/evaluate.py definition), all matches

  python scripts/stream_experiment.py run --variants motion motion+bt det25+bt [--matches ...]
  python scripts/stream_experiment.py report [--variants ...]
Results per (variant, match) are cached in data/exp_stream/results/.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

if not os.environ.get("STREAM_GPU_OCR"):  # EasyOCR on the GPU only when asked (fills the OCR cache fast)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # per-thread BLAS buffers otherwise commit ~1.5 GB per process
import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
from autoscout import config, fuel, opr, pipeline, scouting, tba  # noqa: E402

EXP = config.DATA / "exp_stream"
RESULTS = EXP / os.environ.get("STREAM_RESULTS", "results")  # one directory per scan generation
BLOBS = config.DATA / "exp_fueldet" / "blobs"


# ---------------------------------------------------------------- tracker variants


def track_log(blob_log: dict, fuel_step: int, source: str = "motion", conf: float = 0.25, tracker: str = "greedy",
              split: bool = False, **tk) -> dict:
    """Re-track one match's blob log. source: motion | det | union."""
    out = {}
    for vn, rows in blob_log.items():
        tr = (fuel.FlightTracker(gate=45.0 * fuel_step ** 0.7) if tracker == "greedy"
              else fuel.BallisticTracker(**tk))
        for tc, mb, dd in rows:
            blobs = [] if source == "det" else [fuel.Blob(*map(float, r)) for r in mb]
            if source in ("det", "union"):
                blobs = fuel.merge_detections(blobs, dd, conf)
            tr.update(tc, blobs)
        tracks = tr.flush()
        out[vn] = fuel.split_at_jumps(tracks) if split else tracks
    return out


VARIANTS = {
    "motion": dict(source="motion"),
    "motion+split": dict(source="motion", split=True),
    "motion+bt": dict(source="motion", tracker="ballistic"),
    "motion+bt+split": dict(source="motion", tracker="ballistic", split=True),
    "det25": dict(source="det", conf=0.25),
    "det25+split": dict(source="det", conf=0.25, split=True),
    "det25+bt": dict(source="det", conf=0.25, tracker="ballistic"),
    "det25+bt+split": dict(source="det", conf=0.25, tracker="ballistic", split=True),
    "det15+bt": dict(source="det", conf=0.15, tracker="ballistic"),
    "union25+bt": dict(source="union", conf=0.25, tracker="ballistic"),
    # ablations / sensitivity of the ballistic tracker (motion blobs)
    "motion+bt_k0.4": dict(source="motion", tracker="ballistic", gate_k=0.4),
    "motion+bt_k0.9": dict(source="motion", tracker="ballistic", gate_k=0.9),
    "motion+bt_noseed": dict(source="motion", tracker="ballistic", seed_rmax=0.0),
    "motion+bt_wide": dict(source="motion", tracker="ballistic", gate_min=20.0, gate_k=1.0),
}


# ---------------------------------------------------------------- running


def free_gb() -> float:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "(Get-CimInstance Win32_OperatingSystem).FreeVirtualMemory"], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024 / 1024


def slim(res: dict) -> dict:
    """What scoring and a re-split (data/exp_attrmodel/tools/build_ds.team_signals) need from a result."""
    tl = res["timeline"]
    return {"match": res["match"],
            "timeline": {"auto_start": tl["auto_start"], "teleop_start": tl.get("teleop_start"), "counts": tl.get("counts")},
            "teams": {str(k): v for k, v in res["teams"].items()},
            "paths": res.get("paths", {}),
            "shots": [{"t": s["t"], "team": s["team"], "view": s["view"], "conf": s["conf"], "toward": s.get("toward"),
                       "how": s.get("how")} for s in res["shots"]]}


def cmd_run(keys: list[str], variants: list[str], min_free: float):
    opts = pipeline.Options()
    for key in keys:
        todo = [v for v in variants if not (RESULTS / v / f"{key}.json").exists()]
        if not todo:
            continue
        while free_gb() < min_free:
            time.sleep(30)
        tic = time.time()
        with open(BLOBS / f"{key}.pkl", "rb") as fh:
            log = pickle.load(fh)
        with open(config.OUTPUT / key / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        # identity OCR does not depend on the fuel tracks: read each bumper crop once, keep the reads
        oc_path = EXP / "ocr" / f"{key}.pkl"
        if oc_path.exists():
            with open(oc_path, "rb") as fh:
                md.ocr_cache = {**(getattr(md, "ocr_cache", None) or {}), **pickle.load(fh)}
        n_oc = len(getattr(md, "ocr_cache", None) or {})
        for v in todo:
            md.tracks_all = track_log(log["blob_log"], log["fuel_step"], **VARIANTS[v])
            res = json.loads(json.dumps(pipeline.analyse(md, opts), default=pipeline._json_default))
            d = RESULTS / v
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{key}.json").write_text(json.dumps(slim(res)))
            vp = RESULTS / "_views" / f"{key}.json"
            if not vp.exists():  # which views have a box of each team when (identity does not depend on fuel)
                boxes = defaultdict(lambda: defaultdict(list))
                for row in getattr(md, "det_rows", []):
                    if row[7] is not None:
                        boxes[str(row[7])][row[1]].append(round(float(row[0]), 3))
                vp.parent.mkdir(parents=True, exist_ok=True)
                vp.write_text(json.dumps({t: {v_: sorted(x) for v_, x in d_.items()} for t, d_ in boxes.items()}))
            if len(md.ocr_cache) > n_oc:
                oc_path.parent.mkdir(parents=True, exist_ok=True)
                with open(oc_path, "wb") as fh:
                    pickle.dump(md.ocr_cache, fh, protocol=pickle.HIGHEST_PROTOCOL)
                n_oc = len(md.ocr_cache)
        del md, log
        gc.collect()
        print(f"{key}: {', '.join(todo)} ({time.time() - tic:.0f}s)", flush=True)


# ---------------------------------------------------------------- scoring


def time_valid(row) -> bool:
    return row.time_valid


def burst_table(results: dict[str, dict], bym: dict) -> dict:
    """credited arcs per scouted ball by burst size: {(who, cls): [balls, credited]}"""
    acc = defaultdict(lambda: np.zeros(2))
    for key, r in results.items():
        if key not in bym:
            continue
        info = tba.match(key)
        a0 = r["timeline"]["auto_start"]
        for team in info.red + info.blue:
            srows = [x for x in bym[key].get(team, []) if time_valid(x)]
            if not srows:
                continue
            tt = np.sort([s["t"] for s in r["shots"] if s["team"] == team])
            who = "5026" if team == 5026 else "others"
            w = 1.0 / len(srows)
            for row in srows:
                st = np.sort(a0 + row.shots - 0.8)
                if not len(st):
                    continue
                for b in np.split(st, np.where(np.diff(st) > 1.5)[0] + 1):
                    n = len(b)
                    cls = "1-4" if n < 5 else ("5-9" if n < 10 else ("10-19" if n < 20 else "20+"))
                    c = int(((tt >= b[0] - 1.0) & (tt <= b[-1] + 1.0)).sum()) if len(tt) else 0
                    for k in ((who, cls), ("all", cls), (who, "all"), ("all", "all")):
                        acc[k] += w * np.array([n, c])
    return acc


def scout_share(results: dict[str, dict], bym: dict) -> dict:
    """Share corr / MAE vs mean scouted shots (alliances with all three robots scouted) + 5026 gap."""
    a, b, gap = [], [], []
    for key, r in results.items():
        if key not in bym:
            continue
        info = tba.match(key)
        for al in ("red", "blue"):
            tl = info.teams(al)
            if not all(bym[key].get(t) for t in tl):
                continue
            s = np.array([np.mean([x.n for x in bym[key][t]]) for t in tl])
            v = np.array([r["teams"][str(t)]["fuel"] for t in tl])
            if s.sum() <= 0 or v.sum() <= 0:
                continue
            a += (v / v.sum()).tolist()
            b += (s / s.sum()).tolist()
            if 5026 in tl:
                i = tl.index(5026)
                gap.append(v[i] / v.sum() - s[i] / s.sum())
    a, b = np.array(a), np.array(b)
    slope = float(np.polyfit(b, a, 1)[0]) if len(a) > 3 else float("nan")
    return {"n": len(a), "corr": float(np.corrcoef(a, b)[0, 1]) if len(a) > 3 else float("nan"),
            "mae": float(np.mean(np.abs(a - b))) if len(a) else float("nan"), "slope": slope,
            "gap5026": float(np.mean(gap)) if gap else float("nan"), "n5026": len(gap)}


def load_results(variant: str, keys: list[str] | None = None) -> dict[str, dict]:
    d = RESULTS / variant
    out = {}
    for f in sorted(d.glob("*.json")):
        if keys is None or f.stem in keys:
            out[f.stem] = json.loads(f.read_text())
    return out


def coverage_recall(results: dict[str, dict], bym: dict) -> dict:
    """Failure-finder cycle 11: a scouted ball is detected if a credited toward-hub launch of that team lies
    within +-1 s; split by which views have a box of that team within +-0.15 s of the ball:
    own end view (with or without main) / main only / opposite end only / none. -> {class: [balls, detected]}"""
    acc = defaultdict(lambda: np.zeros(2))
    for key, r in results.items():
        if key not in bym:
            continue
        vp = RESULTS / "_views" / f"{key}.json"
        if not vp.exists():
            continue
        views_ = json.loads(vp.read_text())
        info = tba.match(key)
        a0 = r["timeline"]["auto_start"]
        for al in ("red", "blue"):
            for team in info.teams(al):
                srows = [x for x in bym[key].get(team, []) if time_valid(x)]
                if not srows:
                    continue
                L = np.array([s["t"] for s in r["shots"] if s["team"] == team and s.get("toward")])
                vt = {v_: np.array(x) for v_, x in views_.get(str(team), {}).items()}
                w = 1.0 / len(srows)
                for row in srows:
                    for st in a0 + row.shots - 0.8:
                        det = bool(len(L) and np.min(np.abs(L - st)) <= 1.0)
                        seen = {v_ for v_, x in vt.items() if len(x) and np.min(np.abs(x - st)) <= 0.15}
                        cls = ("own end" if f"{al}_end" in seen else "main only" if "main" in seen
                               else "opp end only" if seen else "none")
                        for c in (cls, "all"):
                            acc[c] += (w, w * det)
    return acc


def scout_rows(results: dict[str, dict], bym: dict) -> dict[str, list]:
    """per match: [(video share, scout share, is_5026)] for alliances with all three robots scouted."""
    out = {}
    for key, r in results.items():
        if key not in bym:
            continue
        info = tba.match(key)
        for al in ("red", "blue"):
            tl = info.teams(al)
            if not all(bym[key].get(t) for t in tl):
                continue
            s = np.array([np.mean([x.n for x in bym[key][t]]) for t in tl])
            v = np.array([r["teams"][str(t)]["fuel"] for t in tl])
            if s.sum() > 0 and v.sum() > 0:
                out.setdefault(key, []).extend(zip(v / v.sum(), s / s.sum(), [t == 5026 for t in tl]))
    return out


def bootstrap(base: dict, var: dict, n: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over matches: base / var = {"opr": {match: team rows}, "scout": {match: rows}}."""
    rng = np.random.default_rng(seed)
    keys = sorted(base["opr"])
    skeys = sorted(base["scout"])
    d_opr, d_sc, d_gap = [], [], []
    for _ in range(n):
        pick = rng.choice(keys, len(keys))
        ra, rb = [], []
        for i, k in enumerate(pick):
            ra += [{**x, "match": f"{k}#{i}"} for x in base["opr"][k]]
            rb += [{**x, "match": f"{k}#{i}"} for x in var["opr"][k]]
        d_opr.append(evaluate.team_metrics(rb)["corr"] - evaluate.team_metrics(ra)["corr"])
        if len(skeys) >= 3:
            sp = rng.choice(skeys, len(skeys))
            a = np.array([x for k in sp for x in base["scout"][k]], float)
            b = np.array([x for k in sp for x in var["scout"][k]], float)
            d_sc.append(np.corrcoef(b[:, 0], b[:, 1])[0, 1] - np.corrcoef(a[:, 0], a[:, 1])[0, 1])
            ga, gb = a[a[:, 2] > 0], b[b[:, 2] > 0]
            if len(ga):
                d_gap.append(np.mean(gb[:, 0] - gb[:, 1]) - np.mean(ga[:, 0] - ga[:, 1]))
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [5, 95])] if len(x) else None
    return {"d_opr_corr_ci90": q(d_opr), "p_opr_better": float(np.mean(np.array(d_opr) > 0)),
            "d_scout_corr_ci90": q(d_sc), "p_scout_better": float(np.mean(np.array(d_sc) > 0)) if d_sc else None,
            "d_gap5026_ci90": q(d_gap), "p_gap_smaller": float(np.mean(np.array(d_gap) > 0)) if d_gap else None}


def cmd_report(variants: list[str], keys: list[str] | None, out_json: Path | None = None):
    bym = scouting.by_match(scouting.load())
    common = None
    for v in variants:
        ks = set(load_results(v, keys))
        common = ks if common is None else common & ks
    common = sorted(common or [])
    oprs = {}
    report = {"matches": common, "variants": {}}
    print(f"{len(common)} matches ({sum(1 for k in common if k in bym)} scouted)\n")
    hdr = (f"{'variant':16s} {'OPR corr':>8s} {'top':>7s} | {'scout corr':>10s} {'mae':>6s} {'slope':>5s} {'5026 gap':>8s} | "
           "credited per scouted ball: 1-4 / 5-9 / 10-19 / 20+ (5026 20+) | arcs/ball all")
    print(hdr)
    for v in variants:
        R = load_results(v, common)
        rows = []
        for k, r in R.items():
            info = tba.match(k)
            if info.event_key not in oprs:
                oprs[info.event_key] = opr.fuel_opr(info.event_key)
            rows += evaluate.team_rows_of(r, info, oprs[info.event_key])
        tm = evaluate.team_metrics(rows)
        ss = scout_share(R, bym)
        bt = burst_table(R, bym)
        per = {f"{w}|{c}": (float(bt[(w, c)][1] / bt[(w, c)][0]) if bt[(w, c)][0] else float("nan"))
               for w in ("all", "5026", "others") for c in ("1-4", "5-9", "10-19", "20+", "all")}
        ev = {e: evaluate.team_metrics([x for x in rows if x["match"].startswith(e + "_")])
              for e in sorted({k.split("_")[0] for k in R})}
        cov = coverage_recall(R, bym)
        covd = {c: {"balls": float(x[0]), "recall": float(x[1] / x[0]) if x[0] else float("nan")} for c, x in cov.items()}
        print(f"{'':16s}   recall per scouted ball by coverage: " + ", ".join(
            f"{c} {covd[c]['recall']:.2f} (n {covd[c]['balls']:.0f})" for c in ("own end", "main only", "opp end only", "none", "all")
            if c in covd))
        report["variants"][v] = {"opr": tm, "scout": ss, "per_ball": per, "by_event": ev, "coverage": covd,
                                 "_rows": {"opr": {k: [x for x in rows if x["match"] == k] for k in R},
                                           "scout": scout_rows(R, bym)}}
        print(f"{v:16s} {tm['corr']:8.3f} {tm['top']:3d}/{tm['alliances']:<3d} | {ss['corr']:10.3f} {ss['mae']:6.3f} "
              f"{ss['slope']:5.2f} {ss['gap5026']:+8.3f} | {per['all|1-4']:.2f} / {per['all|5-9']:.2f} / "
              f"{per['all|10-19']:.2f} / {per['all|20+']:.2f} ({per['5026|20+']:.2f}) | {per['all|all']:.2f}  "
              + "  ".join(f"{e} {m['corr']:.2f}" for e, m in ev.items()))
    base = variants[0]
    print(f"\npaired bootstrap over matches vs {base} (90 % CI of the difference; P(variant better)):")
    for v in variants[1:]:
        b = bootstrap(report["variants"][base]["_rows"], report["variants"][v]["_rows"])
        report["variants"][v]["vs_" + base] = b
        print(f"  {v:18s} OPR corr {b['d_opr_corr_ci90']} P={b['p_opr_better']:.2f} | scout corr {b['d_scout_corr_ci90']} "
              f"P={b['p_scout_better'] if b['p_scout_better'] is None else round(b['p_scout_better'], 2)} | "
              f"5026 gap {b['d_gap5026_ci90']} P(smaller)={b['p_gap_smaller'] if b['p_gap_smaller'] is None else round(b['p_gap_smaller'], 2)}")
    for v in variants:
        report["variants"][v].pop("_rows", None)
    if out_json:
        out_json.write_text(json.dumps(report, indent=1, default=float))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "report"])
    ap.add_argument("--variants", nargs="*", default=["motion", "motion+bt", "det25", "det25+bt"])
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--scouted-cc", action="store_true", help="only cc quals with scouting")
    ap.add_argument("--min-free", type=float, default=4.0, help="GB of free virtual memory before each match")
    ap.add_argument("--out", default=None)
    ap.add_argument("--shard", default="0/1", help="run: i/n, every n-th match starting at i")
    a = ap.parse_args()
    keys = a.matches or sorted(p.stem for p in BLOBS.glob("*.pkl"))
    if a.scouted_cc:
        bym = scouting.by_match(scouting.load())
        keys = [k for k in keys if k.startswith("2026cc_qm") and k in bym]
    if a.cmd == "run":
        si, sn = map(int, a.shard.split("/"))
        cmd_run(keys[si::sn], a.variants, a.min_free)
    else:
        cmd_report(a.variants, a.matches, Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
