"""Compare the bumper-only detector with the whole-robot detector on the same matches.

Configurations (all analysed with the current code, in memory; output/ is never written):
  robot        data/eval_snapshot/<match>/scan.pkl (whole-robot YOLO v1, the frozen baseline)
  robot_v2     data/exp_bumpers/robot_v2/<match>/scan.pkl (same scans re-detected with robots_v2)
  robot_push0  same, without the ROBOT_HALF_DEPTH push from the box edge to the robot centre
  bumper       data/exp_bumpers/b1/<match>/scan.pkl (scripts/rescan_bumpers.py), with the push
  bumper_push0 same, floor point = bumper bottom-centre (no push)

Per configuration:
  * scripts/eval_analysis.py --holdout: identity held-out agreement, share correlation vs fuel
    OPR and top-scorer agreement (2026cc is a test event; train events have no rescans here)
  * tracking: n_segments (fragmentation), mean segment length, coverage (share of 0.5 s match
    bins where a team has a tracked position), robots fused per alliance per step, and cross-
    camera agreement (end-camera fix vs nearest same-alliance main-camera fix at the same time)

Writes data/exp_bumpers/results.json and data/exp_bumpers/RESULTS_section.md (appended to
RESULTS.md by scripts/evaluate.py).

Usage: python scripts/bumper_experiment.py [--exp data/exp_bumpers/b1] [--workers 2] [--configs robot bumper ...]
(results of configurations not re-run are kept from the previous results.json)
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
SNAP = ROOT / "data" / "eval_snapshot"
EXP = ROOT / "data" / "exp_bumpers"
MATCH_LEN = 160.0  # s: auto 20 + teleop 140


def track_metrics(scan: Path, half_depth: float | None) -> dict:
    from autoscout import pipeline, robots

    if half_depth is not None:
        pipeline.ROBOT_HALF_DEPTH = half_depth
    with open(scan, "rb") as fh:
        md = pickle.load(fh)
    pipeline.load_dino(md, scan.parent)
    res = pipeline.analyse(md, pipeline.Options())
    segs = [s for al in ("red", "blue") for s in res["identity"][al].values()]
    dur = [s["t1"] - s["t0"] for s in segs]
    # coverage: 0.5 s match-time bins with a tracked position, per team
    cov = []
    for team in md.info.red + md.info.blue:
        ts = np.array([p[0] for p in res["paths"].get(team, [])])
        bins = np.zeros(int(MATCH_LEN / 0.5), bool)
        if len(ts):
            idx = (ts[(ts >= 0) & (ts < MATCH_LEN)] / 0.5).astype(int)
            bins[idx] = True
        cov.append(bins.mean())
    # cross-camera agreement and robots per alliance per step
    by_t: dict = {}
    for d in md.dets:
        by_t.setdefault(round(d.t, 3), []).append(d)
    dist, n_fused = [], []
    for t, ds in by_t.items():
        fused = robots.fuse(ds)
        for al in ("red", "blue"):
            n_fused.append(sum(f["alliance"] == al for f in fused))
            main = [np.asarray(d.xy) for d in ds if d.view == "main" and d.alliance == al and d.xy is not None]
            if not main:
                continue
            for d in ds:
                if d.view != "main" and d.alliance == al and d.xy is not None:
                    m = min(float(np.hypot(*(np.asarray(d.xy) - x))) for x in main)
                    if m < 2.5:
                        dist.append(m)
    n_fused = np.array(n_fused)
    return {"n_segments": res["n_segments"], "n_dets": res["n_dets"], "seg_s": float(np.mean(dur)) if dur else 0.0,
            "coverage": float(np.mean(cov)), "xview": dist,
            "fused_eq3": float(np.mean(n_fused == 3)), "fused_gt3": float(np.mean(n_fused > 3)),
            "fused_lt3": float(np.mean(n_fused < 3)),
            "n_sides": [getattr(d, "n_sides", 1) for d in md.dets]}


def _tm(args):
    return track_metrics(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default=str(EXP / "b1"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--skip_eval", action="store_true", help="tracking metrics only")
    ap.add_argument("--configs", nargs="*", default=None)
    ap.add_argument("--matches", nargs="*", default=None, help="default: the matches rescanned in --exp")
    a = ap.parse_args()
    exp = Path(a.exp)
    keys = a.matches or sorted(p.name for p in exp.iterdir() if (p / "scan.pkl").exists())
    configs = {"robot": (SNAP, None), "robot_push0": (SNAP, 0.0), "robot_v2": (EXP / "robot_v2", None),
               "bumper": (exp, None), "bumper_push0": (exp, 0.0)}
    if a.configs:
        configs = {k: v for k, v in configs.items() if k in a.configs}
    prev = EXP / "results.json"
    out = json.loads(prev.read_text()) if prev.exists() else {"configs": {}}
    if out.get("matches") not in (None, keys):
        out["configs"] = {}  # different match set: start over
    out["matches"] = keys
    from concurrent.futures import ProcessPoolExecutor

    for name, (snap, hd) in configs.items():
        row = {}
        if not a.skip_eval:
            save = EXP / f"eval_{name}.json"
            cmd = [sys.executable, str(ROOT / "scripts" / "eval_analysis.py"), "--holdout", "--snap", str(snap),
                   "--workers", str(a.workers), "--save", str(save), "--matches", *keys]
            if hd is not None:
                cmd += ["--param", f"pipeline.ROBOT_HALF_DEPTH={hd}"]
            print(name, subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip(), flush=True)
            s = json.loads(save.read_text())["summary"]
            row.update(identity=s.get("identity_holdout"), corr=s["test"]["corr"], top=s["test"]["top"],
                       n_alliances=s["test"]["n"])
        with ProcessPoolExecutor(a.workers) as ex:
            tms = list(ex.map(_tm, [(snap / k / "scan.pkl", hd) for k in keys]))
        xv = np.concatenate([t["xview"] for t in tms]) if any(t["xview"] for t in tms) else np.zeros(0)
        sides = np.concatenate([t["n_sides"] for t in tms])
        row.update(n_segments=float(np.mean([t["n_segments"] for t in tms])),
                   seg_s=float(np.mean([t["seg_s"] for t in tms])),
                   coverage=float(np.mean([t["coverage"] for t in tms])),
                   n_dets=float(np.mean([t["n_dets"] for t in tms])),
                   xview_median=float(np.median(xv)) if len(xv) else None,
                   xview_lt05=float(np.mean(xv < 0.5)) if len(xv) else None, xview_n=int(len(xv)),
                   fused_eq3=float(np.mean([t["fused_eq3"] for t in tms])),
                   fused_gt3=float(np.mean([t["fused_gt3"] for t in tms])),
                   fused_lt3=float(np.mean([t["fused_lt3"] for t in tms])),
                   multi_side=float(np.mean(sides > 1)),
                   per_match_segments={k: t["n_segments"] for k, t in zip(keys, tms)})
        out["configs"][name] = row
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()
                     if k != "per_match_segments"}, flush=True)
    (EXP / "results.json").write_text(json.dumps(out, indent=1))
    print(markdown(out))


def markdown(out: dict) -> str:
    c = out["configs"]
    lines = ["| detector | floor point | identity held-out | share corr vs OPR | top scorer | segments / match | "
             "mean segment (s) | coverage | 3 robots / alliance / step | end vs main camera (median m) |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    label = {"robot": ("whole robot v1", "box edge + 0.4 m"), "robot_push0": ("whole robot v1", "box edge"),
             "robot_v2": ("whole robot v2", "box edge + 0.4 m"),
             "bumper": ("bumpers", "bumper edge + 0.4 m"), "bumper_push0": ("bumpers", "bumper edge")}
    for name in [k for k in label if k in c]:
        r = c[name]
        f = lambda v, p=1, s="": "-" if v is None else f"{v * (100 if s == '%' else 1):.{p}f}{s}"  # noqa: E731
        lines.append(f"| {label[name][0]} | {label[name][1]} | {f(r.get('identity'), 1, '%')} | {f(r.get('corr'), 3)} | "
                     f"{f(r.get('top'), 0, '%')} | {f(r['n_segments'], 1)} | {f(r['seg_s'], 1)} | {f(r['coverage'], 0, '%')} | "
                     f"{f(r['fused_eq3'], 0, '%')} | {f(r['xview_median'], 2)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
