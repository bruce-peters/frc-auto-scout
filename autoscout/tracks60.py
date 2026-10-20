"""60 Hz flight tracks for analysis, rebuilt from a match's 60 Hz blob log.

The scan tracks FUEL at 30 Hz (Options.fuel_step = 2). In a 20 balls/s stream that is too coarse: each ball moves
about as far between analysed frames as the gap to the next ball, so tracks chain one ball onto the next. A
fuel-only re-scan at 60 Hz (Options.fuel_step = 1, data/exp_track/tools/rescan60.py) stores every frame's motion
blobs in config.BLOBS60/<key>.pkl.gz. load() re-runs the greedy flight tracker on them, per view, exactly as
the adopted A/B arm did (data/exp_track/RESULTS_section.md: on 54 scouted quals with today's linking, refit fuel
MAE 41.2 -> 39.8). pipeline.analyse then splits and gates these tracks like the stored ones.

Only for true 60p video. Most FIRST-production webcasts are 30p pictures each shown twice in a 60 fps file (57 of
62 here; every Chezy one and 5 Championship matches are true 60p): tracked on every frame, each ball goes still,
jump, still, jump, split_at_jumps cuts the tracks into 2-position pieces and the launch gate drops them
(2026cancmp_qm5: 1,068 credited launches at 30 Hz -> 142). load() returns None for such a log, so the match keeps
its 30 Hz tracks, which already have one analysed frame per picture (data/exp_track/RESULTS_section.md, "60 Hz on
the FIRST production"; data/exp_track/tools/dup60.py).
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path

import numpy as np

from . import config, fuel

LOG_REPEAT_MAX = 0.30  # share of a frame's blobs repeated (< 1.5 px) from the previous one: doubled 0.40-0.52,
                       # true 60p 0.12-0.19


def log_path(key: str) -> Path | None:
    """The match's 60 Hz blob log (gzip or plain pickle), or None."""
    for name in (f"{key}.pkl.gz", f"{key}.pkl"):
        p = Path(config.BLOBS60) / name
        if p.exists():
            return p
    return None


def tracks_from_log(log: dict) -> dict:
    """{view: [fuel.FlightTrack]} from a blob log {"fuel_step", "blob_log": {view: [(t, blobs, dets)]}}."""
    step = int(log.get("fuel_step", 1))
    out = {}
    for view, rows in log["blob_log"].items():
        tracker = fuel.FlightTracker(gate=45.0 * step ** 0.7)
        for row in rows:
            tracker.update(row[0], [fuel.Blob(*map(float, r)) for r in row[1]])
        out[view] = tracker.flush()
    return out


def log_repeat(log: dict, view: str = "main") -> float:
    """Higher of the two frame phases' share of main-view blobs that reappear within 1.5 px in the next frame."""
    rows = log["blob_log"].get(view) or []
    rep = []
    for a, b in zip(rows[:-1], rows[1:]):
        A, B = np.asarray(a[1])[:, :2], np.asarray(b[1])[:, :2]
        if len(A) and len(B):
            rep.append(float((np.sqrt(((B[:, None] - A[None]) ** 2).sum(-1)).min(1) < 1.5).mean()))
        else:
            rep.append(np.nan)
    rep = np.array(rep)
    if not np.isfinite(rep).any():
        return 0.0
    return float(max(np.nanmean(rep[0::2]), np.nanmean(rep[1::2])))


def duplicated_pictures(log: dict) -> bool:
    """A 60 Hz log of 30p pictures shown twice."""
    return int(log.get("fuel_step", 1)) == 1 and log_repeat(log) > LOG_REPEAT_MAX


def load(key: str) -> dict | None:
    """60 Hz flight tracks of one match, or None when it has no 60 Hz blob log or its video is 30p shown twice."""
    p = log_path(key)
    if p is None:
        return None
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rb") as fh:
        log = pickle.load(fh)
    if duplicated_pictures(log):
        return None
    return tracks_from_log(log)
