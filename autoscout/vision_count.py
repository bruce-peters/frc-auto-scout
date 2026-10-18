"""Vision-only alliance fuel counter (no overlay, no TBA).

Per frame the scanner records moving-FUEL activity in two 3-D regions around each hub,
projected into every calibrated view: the airspace over the opening ("top") and the floor in
front of the four exits ("exit", where every counted ball leaves the hub). These are binned
per second and mapped to balls/second by a non-negative linear model fitted against the
overlay counter on matches that have one (the overlay is exact ground truth per second).
Fuel is only counted while the alliance's hub is active.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from . import config

BIN = 1.0
VIEWS = ("main", "near_end", "far_end")  # end cameras relative to the alliance's own hub
KINDS = ("top", "exit")
MODEL_PATH = config.MODELS / "vision_count.json"


def feature_names() -> list[str]:
    return [f"{v}_{k}_{q}" for v in VIEWS for k in KINDS for q in ("n", "area")]


def _view_key(view: str, alliance: str) -> str:
    if view == "main":
        return "main"
    return "near_end" if view.startswith(alliance) else "far_end"


def load_feats(match_key: str) -> dict:
    p = config.OUTPUT / match_key / "hub_feats.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    return {tuple(k.split("|")): np.array(v, dtype=np.float64).reshape(-1, 3) for k, v in raw.items()}


def binned(feats: dict, alliance: str, t0: float, t1: float, own_end: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin start times, X[n_bins, n_features]) for one alliance's hub."""
    edges = np.arange(t0, t1 + BIN, BIN)
    X = np.zeros((len(edges) - 1, len(feature_names())))
    col = 0
    rev = {_view_key(v, alliance): v for v in ("main", "blue_end", "red_end")}
    for v in VIEWS:
        for k in KINDS:
            arr = feats.get((rev[v], alliance, k))
            if arr is not None and len(arr):
                n, _ = np.histogram(arr[:, 0], edges, weights=arr[:, 1])
                a, _ = np.histogram(arr[:, 0], edges, weights=arr[:, 2])
                # normalise: frames per bin differ with fps / sub-sampling
                cnt, _ = np.histogram(arr[:, 0], edges)
                cnt = np.maximum(cnt, 1)
                X[:, col] = n / cnt
                X[:, col + 1] = a / cnt / 100.0
            col += 2
    return edges[:-1], X


def smooth(X: np.ndarray, w: int = 3) -> np.ndarray:
    if w <= 1:
        return X
    k = np.ones(w) / w
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, X)


def target_from_counts(counts: list, edges_t: np.ndarray, lag: float) -> np.ndarray:
    """Balls per bin from the overlay counter, shifted back by the counter lag."""
    y = np.zeros(len(edges_t))
    prev = 0
    for t, c in counts:
        if c > prev:
            i = int(np.floor((t - lag - edges_t[0]) / BIN))
            if 0 <= i < len(y):
                y[i] += c - prev
        prev = max(prev, c)
    return y


def fit(samples: list[tuple[np.ndarray, np.ndarray]], ridge: float = 1e-3) -> np.ndarray:
    X = np.vstack([s[0] for s in samples])
    y = np.concatenate([s[1] for s in samples])
    Xa = np.hstack([X, np.ones((len(X), 1))])
    # tiny ridge via augmentation, non-negative weights (more activity never means fewer balls)
    Xr = np.vstack([Xa, np.sqrt(ridge) * np.eye(Xa.shape[1])])
    yr = np.concatenate([y, np.zeros(Xa.shape[1])])
    w, _ = nnls(Xr, yr)
    return w


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def save(w: np.ndarray, meta: dict):
    MODEL_PATH.write_text(json.dumps({"w": w.tolist(), "features": feature_names(), **meta}, indent=1))


def load() -> tuple[np.ndarray, dict] | None:
    if not MODEL_PATH.exists():
        return None
    d = json.loads(MODEL_PATH.read_text())
    return np.array(d["w"]), d
