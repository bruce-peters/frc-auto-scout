"""Diagnostic main camera with the right 3-D geometry: the hand-checked Chezy landmark camera
(data/exp_maincal/prior_lm.json: tags + two field-line pixels on 2026cc_qm3; f 1423 px, centre (7.8, 21.5, 6.5) m
instead of the tag-only fits' ~2140 px and (8.2, 28.7, 11.9) m), posed per match by rotation only about its fixed
centre so it reproduces the stored camera's image of the main view's tags (2, 11, 21, 24).

Not zero-context (the landmarks were picked by hand once per event): used only to test whether the main camera's
3-D calibration is what stops cross-view pairs from fitting.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, field as F, maincal  # noqa: E402

PRIOR = config.DATA / "exp_maincal" / "prior_lm.json"
MAIN_TAGS = (2, 11, 21, 24)


def lm_main_track(cam_track, prior_path=PRIOR):
    prior = json.loads(Path(prior_path).read_text())
    tags = F.load_tags()
    W, H = prior["size"]
    obs = []
    for t in cam_track.times:
        cam = cam_track.at(float(t))
        det = {}
        for i in MAIN_TAGS:
            if i not in tags:
                continue
            uv = cam.project(F.tag_corners(tags[i]))
            if np.all((uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)):
                det[i] = uv
        if det:
            obs.append((float(t), det))
    return maincal.static_from_prior(obs, tags, prior, max_frame_err=6.0)
