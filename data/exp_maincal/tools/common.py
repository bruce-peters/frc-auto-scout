"""Shared helpers for the main-camera experiments (data/exp_maincal/tools)."""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # never touch the GPU
ROOT = Path(r"C:\Users\bruce\Documents\Coding\auto-scout")
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from autoscout import field as F, maincal as M, pipeline as P, robots, views  # noqa: E402

EXP = ROOT / "data" / "exp_maincal"
EXTRACT = EXP / "extract"
VN = ("main", "blue_end", "red_end")
TAGS = F.load_tags()


def load_extract(key: str) -> dict:
    with open(EXTRACT / f"{key}.pkl", "rb") as fh:
        return pickle.load(fh)


def keys(prefix: str = "") -> list[str]:
    return sorted(p.stem for p in EXTRACT.glob(f"{prefix}*.pkl"))


def main_boxes(ex: dict) -> np.ndarray:
    """(t, u, v) bottom-centres of main-view boxes in view pixels."""
    d = ex["dets"]
    d = d[d[:, 1] == 0]
    v = next(x for x in views.PRODUCTIONS[ex["prod"]] if x.name == "main")
    ox, oy = v.offset(1920, 1080)
    return np.stack([d[:, 0], (d[:, 2] + d[:, 4]) / 2 - ox, d[:, 5] - oy], axis=1)


def fake_md(ex: dict, main_cam=None):
    """Enough of a MatchData for pipeline.cross_view_error."""
    dets = [robots.Detection(float(r[0]), VN[int(r[1])], tuple(r[2:6]), float(r[7]), "red" if r[6] else "blue")
            for r in ex["dets"]]
    cams = dict(ex["cams"])
    if main_cam is not None:
        cams["main"] = main_cam
    return SimpleNamespace(cams=cams, prod=ex["prod"], dets=dets)


def xview(md, cam_main) -> dict:
    """cross_view_error of both end views with `cam_main` as the main camera."""
    old = md.cams.get("main")
    md.cams["main"] = cam_main
    out = {}
    for v in ("blue_end", "red_end"):
        if v in md.cams:
            g = P.cross_view_error(md, v, md.cams[v])
            out[v] = (round(g["median"], 3), g["n"])
    md.cams["main"] = old
    return out


def off_share(cam, boxes) -> float:
    return float(M.off_field(cam, boxes, P.ROBOT_HALF_DEPTH).mean()) if len(boxes) else float("nan")


# hand-picked: where the white centre line meets the far / near wall base (2026cc_qm3, t 8 s, main view pixels)
MANUAL_PTS = (np.array([[F.config.FIELD_LENGTH / 2, 0.0, 0.0], [F.config.FIELD_LENGTH / 2, F.config.FIELD_WIDTH, 0.0]]),
              np.array([[943.0, 391.0], [924.0, 617.0]]))
