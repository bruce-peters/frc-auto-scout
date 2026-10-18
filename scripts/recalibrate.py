"""Recover missing end-camera calibrations in saved scans, without a rescan.

For each match, every end view missing from md.cams is calibrated again from data/videos/<match>.mp4
(the same frames as pipeline.calibrate, every 4 s) with the new logic:

1. the plain tag fit (field.calibrate_view_dynamic), as in the scan;
2. with the event prior - the median K, distortion and camera centre of that view over the event's other
   matches, data/cache/calib_<event>_<view>.json - field.recover_view adds a fit started from the event K
   (focal length within 30 % of the event median) and a fixed-intrinsics pan/tilt track (solvePnPRansac
   where >= 2 tags show, rotation about the camera centre where only 1 does);
3. pipeline.accept_recovered_cams keeps the candidate whose robots best agree with the main view's
   (median distance to the nearest same-alliance main-view robot at the same instant < 1 m), else none.

The scan is saved with pipeline._save_scan (atomically replacing scan.pkl) and a calib_recovery.json
report is written beside it. pipeline.analyse recomputes every field position from md.cams, so a
reanalyse (or finalize) afterwards picks the camera up.

The event prior is seeded first from the scans in --seed (default: <scan_dir>) of the same events; matches
already seen are skipped, so later runs are quick.

  .venv/Scripts/python scripts/recalibrate.py output 2026cc_qm15 2026cagle_qm32
  .venv/Scripts/python scripts/recalibrate.py output --missing      (every match in output/ missing an end view)
  options: --seed DIR [DIR ...]   --dry-run (report only)   --views blue_end red_end
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoscout import config, field as F, pipeline as P, views  # noqa: E402
from autoscout.video import Video  # noqa: E402

END_VIEWS = ("blue_end", "red_end")


def _load(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def seed_priors(events: set[str], dirs: list[Path], end_views=END_VIEWS) -> None:
    """Add every saved scan's plain end-view fits of these events to the event priors."""
    for ev in sorted(events):
        seen = set.intersection(*[set((F._read_json(F.calib_prior_path(ev, v)) or {}).get("seen", []))
                                  for v in end_views])
        todo = [p for d in dirs for p in sorted(Path(d).glob(f"{ev}_*/scan.pkl")) if p.parent.name not in seen]
        if todo:
            print(f"seeding the {ev} prior from {len(todo)} scans ...", flush=True)
        for p in todo:
            md = _load(p)
            for v in end_views:
                F.update_calib_prior(ev, v, p.parent.name, md.cams.get(v))
            del md
            gc.collect()
        for v in end_views:
            pr = F.load_calib_prior(ev, v)
            print(f"  prior {ev} {v}: " + ("none (too few calibrated matches)" if pr is None else
                  f"{pr['n']} matches, f {pr['f']:.0f} px, k1 {pr['dist'][0]:+.3f}, centre "
                  f"{np.round(pr['centre'], 2).tolist()}"), flush=True)


def _summary(pr: dict | None) -> dict | None:
    if pr is None:
        return None
    return {"n": pr["n"], "f": round(pr["f"], 1), "k1": round(pr["dist"][0], 4),
            "centre": np.round(pr["centre"], 3).tolist()}


def recalibrate(md, key: str, end_views=END_VIEWS, audit: bool = False) -> dict:
    """Recover md's missing end views in place. With `audit`, end views that did calibrate but whose robots
    disagree with the main view (cross_view_error median >= CALIB_GATE_M) are re-solved too; the current
    camera competes as a candidate, so it is only replaced by one that passes the gate and agrees better.
    Returns the per-view report."""
    event = md.info.event_key
    todo = [v for v in views.PRODUCTIONS[md.prod] if v.name in end_views and v.name not in md.cams]
    suspect = {}
    if audit:
        for v in views.PRODUCTIONS[md.prod]:
            if v.name in end_views and v.name in md.cams:
                g = P.cross_view_error(md, v.name, md.cams[v.name])
                if g["n"] >= P.CALIB_GATE_MIN_N and g["median"] >= P.CALIB_GATE_M:
                    todo.append(v)
                    suspect[v.name] = {k: round(float(x), 3) if k != "n" else x for k, x in g.items()}
    out = {"match": key, "event": event, "views": {}, "suspect": suspect}
    if not todo:
        return out
    video = Video(config.VIDEOS / f"{key}.mp4")
    tl = md.timeline
    a0 = tl.auto_start if tl.auto_start is not None else P.FIRST_AUTO_START
    tele = tl.teleop_start or (a0 + P.FIRST_TELEOP_GAP)
    times = [float(t) for t in np.arange(a0 - 1, tele + config.TELEOP_LEN + 2, 4.0)]  # as process_match
    frames = [(t, video.frame_at(t)) for t in times]
    video.close()
    frames = [(t, f) for t, f in frames if f is not None]
    times = [t for t, _ in frames]
    tags = F.load_tags()
    for view in todo:
        crops = [view.crop(f) for _, f in frames]
        masks = view.mask_boxes(video.width, video.height)
        tic = time.time()
        cands = [("current", md.cams[view.name])] if view.name in suspect else []
        plain = F.calibrate_view_dynamic(crops, times, tags, masks)
        if plain is not None and plain.rms < F.MAX_RMS:
            cands.append(("tags", plain))
        prior = F.load_calib_prior(event, view.name, crops[0].shape[1::-1], exclude=key)
        diag = {}
        if prior is not None:
            cands += F.recover_view(crops, times, tags, masks, prior, diag=diag)
        rep = P.accept_recovered_cams(md, {view.name: cands})[view.name]
        rep.update(prior=_summary(prior), diag=diag, plain_fit=None if plain is None else
                   {"rms": round(plain.rms, 3), "f": round(plain.K[0][0], 1), "poses": len(plain.times)},
                   seconds=round(time.time() - tic, 1))
        out["views"][view.name] = rep
    return out


def _save(md, match_dir: Path) -> None:
    """pipeline._save_scan into a temporary folder, then replace scan.pkl in one step."""
    tmp = match_dir / ".recalibrate_tmp"
    tmp.mkdir(exist_ok=True)
    P._save_scan(md, tmp)
    os.replace(tmp / "scan.pkl", match_dir / "scan.pkl")
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scan_dir")
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--missing", action="store_true", help="every match in scan_dir whose result.json lacks an end view")
    ap.add_argument("--seed", nargs="*", default=None, help="scan dirs for the event prior (default: scan_dir)")
    ap.add_argument("--views", nargs="*", default=list(END_VIEWS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit", action="store_true",
                    help="also re-solve calibrated end views that disagree with the main view (all matches if none given)")
    a = ap.parse_args()
    scan_dir = Path(a.scan_dir)
    keys = list(a.matches)
    if a.audit and not keys:
        keys = [p.parent.name for p in sorted(scan_dir.glob("*/scan.pkl"))]
    if a.missing:
        for p in sorted(scan_dir.glob("*/result.json")):
            cal = json.loads(p.read_text()).get("calibration", {})
            if any(v not in cal for v in a.views) and p.parent.name not in keys:
                keys.append(p.parent.name)
    keys = [k for k in keys if (scan_dir / k / "scan.pkl").exists()]
    if not keys:
        print("no scans to recalibrate")
        return
    seed_priors({k.split("_")[0] for k in keys}, [Path(d) for d in (a.seed or [scan_dir])])
    for key in keys:
        md = _load(scan_dir / key / "scan.pkl")
        rep = recalibrate(md, key, tuple(a.views), audit=a.audit)
        if not rep["views"]:
            print(f"{key}: every end view calibrated{' and agrees with the main view' if a.audit else ''}", flush=True)
        for v, r in rep["views"].items():
            c = " | ".join(f"{x['how']}: rms {x['rms']:.2f} px, {x['poses']} poses, f {x['f']:.0f}, cross-view "
                           f"median {x['median_m']:.2f} m (n {x['n']}){' PASS' if x['pass'] else ''}"
                           for x in r["candidates"]) or "no candidate"
            print(f"{key} {v}: {'recovered (' + r['accepted'] + ')' if r['accepted'] else 'still uncalibrated'}"
                  f"  [{c}]", flush=True)
        if any(r["accepted"] for r in rep["views"].values()) and not a.dry_run:
            _save(md, scan_dir / key)
            (scan_dir / key / "calib_recovery.json").write_text(json.dumps(rep, indent=1, default=float),
                                                               encoding="utf-8")
            print(f"{key}: saved {scan_dir / key / 'scan.pkl'}", flush=True)
        del md
        gc.collect()


if __name__ == "__main__":
    main()
