"""Re-run only the robot detection of frozen match scans with the bumper-only detector.

Loads data/eval_snapshot/<match>/scan.pkl (camera calibration, overlay timeline, fuel flight
tracks, hub features), replaces its whole-robot detections by bumper detections exactly as
pipeline.scan_video does with Options.bumper_model (autoscout/bumpers.py), re-runs the analysis
and writes scan.pkl / result.json / dets.json to <out>/<match>/. Everything except the robot
detections is reused untouched, so differences against the snapshot come from the detector.
output/ and data/eval_snapshot/ are not modified.

With --robot_model the same frozen scans are re-detected with a whole-robot model instead
(scripts/rescan_robots.detect_robots), for a like-for-like whole-robot baseline.

Usage:
  python scripts/rescan_bumpers.py --model runs/bumpers_v1/weights/best.pt --out data/exp_bumpers/b1 --shard 0/2
  python scripts/rescan_bumpers.py --robot_model models/robots_v2.pt --out data/exp_bumpers/robot_v2
  python scripts/bumper_experiment.py --exp data/exp_bumpers/b1
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from autoscout import bumpers, fuel, pipeline  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402

SNAP = ROOT / "data" / "eval_snapshot"


def detect(md: pipeline.MatchData, video: Video, det: bumpers.BumperDetector, opts: pipeline.Options):
    """The robot block of pipeline.scan_video with a bumper model, alone: fills md.dets / md.crops."""
    det_every = max(1, int(round(video.fps / opts.det_hz)))
    step = max(1, opts.fuel_step)
    start, end = int(pipeline.md_t0(md) * video.fps), int(pipeline.md_t1(md) * video.fps)
    md.dets, md.crops, crop_id = [], {}, 0
    for idx, frame in video.iter_frames(start, end, step=det_every):
        t = idx / video.fps
        # scan_video has this frame's fuel mask buffered only on fuel frames
        yellow = fuel.yellow_mask(frame) if (idx - start) % step == 0 else None
        crop_id = bumpers.scan_frame(md, frame, t, det, crop_id, (idx // det_every) % 2 == 0,
                                     yellow=yellow, end_view_range=pipeline.END_VIEW_RANGE)


def rescan(key: str, det, out_root: Path, model: str, whole_robot: bool = False) -> dict:
    tic = time.time()
    with open(SNAP / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    if whole_robot:
        sys.path.insert(0, str(ROOT / "scripts"))
        from rescan_robots import detect_robots

        opts = pipeline.Options(robot_model=model, out_dir=out_root)
    else:
        opts = pipeline.Options(bumper_model=model, out_dir=out_root)
    video = Video(video_path(key))
    (detect_robots if whole_robot else detect)(md, video, det, opts)
    video.close()
    # crop ids changed: OCR reads and appearance embeddings are redone
    md.ocr_cache, md.cnn_cache, md.dino_cache = {}, None, None
    res = pipeline.analyse(md, opts)
    res["runtime_s"] = round(time.time() - tic, 1)
    res["robot_model" if whole_robot else "bumper_model"] = model
    out_dir = out_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline._save_scan(md, out_dir)
    (out_dir / "dets.json").write_text(json.dumps(getattr(md, "det_rows", []), default=pipeline._json_default))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=pipeline._json_default))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="bumper model")
    ap.add_argument("--robot_model", help="whole-robot model instead (baseline)")
    ap.add_argument("--out", default=str(ROOT / "data" / "exp_bumpers" / "b1"))
    ap.add_argument("--matches", nargs="*", default=None, help="default: every 2026cc match in the snapshot")
    ap.add_argument("--conf", type=float, default=pipeline.Options.bumper_conf)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th match starting at i")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if not (a.model or a.robot_model):
        ap.error("--model (bumpers) or --robot_model (whole robots) is required")
    keys = a.matches or sorted(p.name for p in SNAP.iterdir() if p.name.startswith("2026cc_"))
    out_root = Path(a.out)
    si, sn = map(int, a.shard.split("/"))
    det = None
    for key in keys[si::sn]:
        if not a.force and (out_root / key / "result.json").exists():
            print("skip", key, flush=True)
            continue
        if det is None:
            det = (pipeline.RobotDetector(a.robot_model, pipeline.Options.det_imgsz, pipeline.Options.det_conf)
                   if a.robot_model else bumpers.BumperDetector(a.model, pipeline.Options.det_imgsz, a.conf))
        try:
            res = rescan(key, det, out_root, a.robot_model or a.model, whole_robot=bool(a.robot_model))
            print(f"ok {key} {res['runtime_s']:.0f}s dets={res['n_dets']} segs={res['n_segments']}", flush=True)
        except Exception as e:  # keep going: one bad video must not stop the batch
            print(f"FAIL {key}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
