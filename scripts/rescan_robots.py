"""Re-run only the robot detector over cached match scans, to evaluate a detector change.

Loads output/<match>/scan.pkl (camera calibration, overlay timeline, fuel flight tracks, hub
features), re-detects robots at det_hz exactly as pipeline.scan_video does, re-runs the analysis
(tracking, OCR + appearance identity, attribution) and writes result.json / dets.json / scan.pkl
to <out>/<match>/. Everything except the robot detections is reused untouched, so differences
between two runs with different --model come from the detector alone. output/ is not modified.

Usage:
  python scripts/rescan_robots.py --model runs/robots_v2/weights/best.pt --out data/exp_robots_v2/v2 \
      --matches 2026cc_qm5 2026joh_qm3 --shard 0/3
Then: python scripts/identity_holdout.py --out data/exp_robots_v2/v2
      python scripts/evaluate.py --out data/exp_robots_v2/v2
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, fuel, identity, ocr, pipeline, robots, views  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402


def detect_robots(md: pipeline.MatchData, video: Video, det: pipeline.RobotDetector, opts: pipeline.Options):
    """The robot block of pipeline.scan_video, alone: fills md.dets and md.crops."""
    vlist = views.PRODUCTIONS[md.prod]
    W, H = video.width, video.height
    det_every = max(1, int(round(video.fps / opts.det_hz)))
    step = max(1, opts.fuel_step)
    start, end = int(pipeline.md_t0(md) * video.fps), int(pipeline.md_t1(md) * video.fps)
    md.dets, md.crops, crop_id = [], {}, 0
    for idx, frame in video.iter_frames(start, end, step=det_every):
        t = idx / video.fps
        yellow = None
        for box, conf, al_cls in det(frame):
            v = pipeline._view_of(box, vlist, W, H)
            if v is None:
                continue
            x0, y0, x1, y1 = [int(round(c)) for c in box]
            crop = frame[max(0, y0):y1, max(0, x0):x1]
            al_col, col_conf = robots.bumper_alliance(crop)
            al = al_col if (al_col and col_conf > 0.75) else (al_cls or al_col)
            if al is None:
                continue
            d = robots.Detection(t, v.name, box, conf, al)
            if (idx - start) % step == 0:  # scan_video has this frame's fuel mask buffered
                if yellow is None:
                    yellow = fuel.yellow_mask(frame)
                ym = yellow[max(0, y0):y1, max(0, x0):x1]
                d.fill = float(ym.mean()) if ym.size else None
            # d.xy is recomputed by pipeline.analyse from the calibrated cameras
            bh = y1 - y0
            if bh >= 28 and (idx // det_every) % 2 == 0:
                d.crop_id = crop_id
                md.crops[crop_id] = (ocr.bumper_crop(frame, box), identity.appearance_vector(crop), bh,
                                     cv2.resize(crop, (96, 96)) if crop.size else None)
                crop_id += 1
            md.dets.append(d)


def rescan(key: str, model: str, out_root: Path, det: pipeline.RobotDetector | None = None) -> dict:
    tic = time.time()
    with open(config.OUTPUT / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    opts = pipeline.Options(robot_model=model, out_dir=out_root)
    det = det or pipeline.RobotDetector(opts.robot_model, opts.det_imgsz, opts.det_conf)
    video = Video(video_path(key))
    detect_robots(md, video, det, opts)
    video.close()
    # crop ids changed: OCR / embeddings must be redone (DINO features, scripts/build_dino.py, are
    # per crop id too; without a dino.npz for the new crops, analyse falls back to ResNet-18)
    md.ocr_cache, md.cnn_cache, md.dino_cache = {}, None, None
    res = pipeline.analyse(md, opts)
    res["runtime_s"] = round(time.time() - tic, 1)
    res["robot_model"] = model
    out_dir = out_root / key
    out_dir.mkdir(parents=True, exist_ok=True)
    pipeline._save_scan(md, out_dir)
    (out_dir / "dets.json").write_text(json.dumps(getattr(md, "det_rows", []), default=pipeline._json_default))
    (out_dir / "result.json").write_text(json.dumps(res, indent=1, default=pipeline._json_default))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output root, e.g. data/exp_robots_v2/v2")
    ap.add_argument("--matches", nargs="+", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th match starting at i")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    out_root = Path(a.out)
    si, sn = map(int, a.shard.split("/"))
    det = None
    for key in a.matches[si::sn]:
        if not a.force and (out_root / key / "result.json").exists():
            print("skip", key, flush=True)
            continue
        if det is None:
            o = pipeline.Options()
            det = pipeline.RobotDetector(a.model, o.det_imgsz, o.det_conf)
        try:
            res = rescan(key, a.model, out_root, det)
            n = len(json.loads((out_root / key / "dets.json").read_text()))
            print(f"ok {key} {res['runtime_s']:.0f}s dets={n}", flush=True)
        except Exception as e:  # keep going: one bad video must not stop the batch
            print(f"FAIL {key}: {e!r}", flush=True)


if __name__ == "__main__":
    main()
