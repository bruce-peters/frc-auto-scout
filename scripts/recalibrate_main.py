"""Re-solve the main camera of saved scans with the static model (autoscout/maincal.py), without a rescan.

Failure-finder cycle 8: 30 % of Chezy Champs main-view robot boxes had no field position. Two causes:

1. the Chezy main camera's rotation is ~180 deg, where every rotation has two Rodrigues vectors; the per-window
   fits returned either, and CameraTrack.at blended neighbouring windows across the two, giving garbage poses
   between windows. Fixed in field.CameraTrack.at, so every saved scan is fixed by a plain reanalyse;
2. each 4 s window pose is fitted on its own to four ~12 px coplanar tags, so the camera centre wanders by
   0.5-1.6 m between windows, and the focal length / distance differ from match to match.

This script handles 2, for main views whose stored window centres scatter by more than maincal.MAIN_SCATTER_M
(0.4 m; every non-Chezy main view checked is under 0.28 m and is left alone):

* pass 1, per match: the main view's tags are detected again in data/videos/<match>.mp4 (pipeline.calibrate's
  frames, every 4 s) and maincal.fit_static fits one camera centre, focal length and distortion for the match
  with one rotation per window (started from the stored camera and from the planar-PnP solutions). The fit is
  recorded in the event's main prior, data/cache/calib_<event>_main.json;
* pass 2, per match: the candidates are that static fit and, with an event prior, the rotation-only track about
  the event camera (preferred). The event camera is the median of the other matches' static fits, or with
  --landmarks (or data/cache/calib_<event>_main_landmarks.json) a camera checked against hand-picked field
  points. pipeline.accept_main_cam cleans every candidate with maincal.guard and keeps one only if the share
  of main-view boxes with no field position drops (the preferred one: does not rise) and the end cameras
  agree with it no worse (cross_view_error; with a landmark camera only reported). A match where most window
  poses put their boxes off the field is flagged.

Accepted scans are saved with pipeline._save_scan into a temporary folder and os.replace'd over scan.pkl, with
maincal_recovery.json beside them; pipeline.analyse recomputes every field position from md.cams, so a
reanalyse (or finalize) afterwards picks the camera up. A second run leaves fixed scans alone (--force redoes).

  .venv/Scripts/python scripts/recalibrate_main.py output --event 2026cc
  .venv/Scripts/python scripts/recalibrate_main.py output 2026cc_qm26 2026cc_qm5
  options: --dry-run (report only)   --landmarks FILE   --no-prior   --force
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
from autoscout import config, field as F, maincal, pipeline as P, views  # noqa: E402
from autoscout.video import Video  # noqa: E402


def _load(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def main_observations(md, key: str):
    """(tag sightings, view size) of the main view at pipeline.calibrate's sample times."""
    video = Video(config.VIDEOS / f"{key}.mp4")
    tl = md.timeline
    a0 = tl.auto_start if tl.auto_start is not None else P.FIRST_AUTO_START
    tele = tl.teleop_start or (a0 + P.FIRST_TELEOP_GAP)
    times = [float(t) for t in np.arange(a0 - 1, tele + config.TELEOP_LEN + 2, 4.0)]  # as process_match
    view = next(v for v in views.PRODUCTIONS[md.prod] if v.name == "main")
    masks = view.mask_boxes(video.width, video.height)
    crops, ts = [], []
    for t in times:
        f = video.frame_at(t)
        if f is not None:
            crops.append(view.crop(f))
            ts.append(t)
    video.close()
    if not crops:
        return [], None
    return maincal.tag_observations(crops, ts, masks), crops[0].shape[1::-1]


def needs_refit(md, force: bool = False) -> str | None:
    """Why this scan's main camera is refitted, or None."""
    cam = md.cams.get("main")
    if cam is None:
        return None
    if getattr(cam, "main_fix", None) is not None and not force:
        return None  # done before
    sc = maincal.centre_scatter(cam)
    return f"window centres scatter {sc:.2f} m" if (sc > maincal.MAIN_SCATTER_M or force) else None


def _save(md, match_dir: Path) -> None:
    """pipeline._save_scan into a temporary folder, then replace scan.pkl in one step."""
    tmp = match_dir / ".recalibrate_main_tmp"
    tmp.mkdir(exist_ok=True)
    P._save_scan(md, tmp)
    os.replace(tmp / "scan.pkl", match_dir / "scan.pkl")
    shutil.rmtree(tmp, ignore_errors=True)


def _cand_line(c: dict) -> str:
    return (f"{c['how']}: off {c['off']:.3f} xv {c['xview']} f {c['f']:.0f} rms {c['rms']:.2f} "
            f"dropped {c['dropped']}/{c['poses']}{' PASS' if c['pass'] else ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("scan_dir")
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--event", action="append", default=[], help="every match of this event in scan_dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--landmarks", default=None, help="checked event camera JSON (default: "
                    "data/cache/calib_<event>_main_landmarks.json if it exists)")
    ap.add_argument("--no-prior", action="store_true", help="no event camera: static fits only")
    ap.add_argument("--force", action="store_true", help="also refit views that look fine or were fixed before")
    a = ap.parse_args()
    scan_dir = Path(a.scan_dir)
    keys = list(a.matches)
    for ev in a.event:
        keys += [p.parent.name for p in sorted(scan_dir.glob(f"{ev}_*/scan.pkl")) if p.parent.name not in keys]
    keys = [k for k in keys if (scan_dir / k / "scan.pkl").exists() and (config.VIDEOS / f"{k}.mp4").exists()]
    if not keys:
        print("no scans (with a video) to recalibrate")
        return
    tags = F.load_tags()
    # pass 1: static fit per match (recorded in the event's main prior)
    work = {}
    for key in keys:
        tic = time.time()
        md = _load(scan_dir / key / "scan.pkl")
        why = needs_refit(md, a.force)
        if why is None:
            print(f"{key}: main camera left as it is (window centres agree, or fixed before)", flush=True)
        else:
            obs, size = main_observations(md, key)
            static = dict(maincal.main_candidates(obs, tags, size, md.cams["main"])).get("static")
            if static is not None:
                maincal.update_main_prior(md.info.event_key, key, static)
            work[key] = (obs, size, static, why, md.info.event_key)
            print(f"{key}: {why}; static fit " + ("failed" if static is None else
                  f"f {static.K[0][0]:.0f}, centre {static.static['centre']}, tag rms {static.rms:.2f} px") +
                  f" ({time.time() - tic:.0f}s)", flush=True)
        del md
        gc.collect()
    # pass 2: candidates + gate
    for key, (obs, size, static, why, event) in work.items():
        md = _load(scan_dir / key / "scan.pkl")
        prior, trusted = (None, False) if a.no_prior else maincal.load_main_prior(event, size, exclude=key, path=a.landmarks)
        if prior is not None and prior.get("event") not in (None, event):
            prior, trusted = None, False  # another event's camera
        cands = [("static", static)] if static is not None else []
        if prior is not None and obs:
            ev_tr = maincal.static_from_prior(obs, tags, prior)
            if ev_tr is not None and ev_tr.rms < F.MAX_RMS:
                cands.append(("event_prior", ev_tr))
        rep = P.accept_main_cam(md, cands, prefer="event_prior", trusted=trusted)
        rep.update(match=key, reason=why, frames_with_tags=len(obs),
                   prior=None if prior is None else {"trusted": trusted, "n": prior.get("n"), "f": round(float(prior["K"][0][0]), 1),
                                                     "centre": np.round(prior["centre"], 3).tolist()})
        cur = rep.get("current") or {}
        print(f"{key}: {'replaced (' + rep['accepted'] + ')' if rep.get('accepted') else 'kept the stored camera'}"
              f"{'  FLAG: most window poses put their boxes off the field' if rep.get('flag') else ''}\n"
              f"    current: off {cur.get('off', float('nan')):.3f} xv {cur.get('xview')}\n    " +
              "\n    ".join(_cand_line(c) for c in rep.get("candidates", [])), flush=True)
        if rep.get("accepted") and not a.dry_run:
            _save(md, scan_dir / key)
            (scan_dir / key / "maincal_recovery.json").write_text(json.dumps(rep, indent=1, default=float),
                                                                  encoding="utf-8")
            print(f"{key}: saved {scan_dir / key / 'scan.pkl'}", flush=True)
        del md
        gc.collect()


if __name__ == "__main__":
    main()
