"""Merge YOLO datasets into one, remapping classes, converting polygons to boxes, deduplicating.

Splits are re-drawn from a hash of each image's original stem, so the same frame exported by
different Roboflow forks can never land in both train and val.

Usage:
  python scripts/merge_datasets.py robots   -> data/datasets/robots_v1 (single class 'robot')
  python scripts/merge_datasets.py fuel     -> data/datasets/fuel_v1   (single class 'fuel')
  python scripts/merge_datasets.py robots --name robots_v2   -> data/datasets/robots_v2
  python scripts/merge_datasets.py bumpers  -> data/datasets/bumpers_v1 (red_bumper / blue_bumper)
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "data" / "datasets"

# source dir -> {source class name: target class id}; unknown names are dropped
RECIPES = {
    "robots": {
        "bumper-based-detection-om4u6": {"robot_blue": 0, "robot_red": 0},
        "new-5419-cv-d6qtg": {"robot": 0},
        "robot-seg-06wv0": {"red robot": 0, "blue robot": 0},
        "frc-robot-id": {"FRC-Robot-red": 0, "FRC-Robot-blue": 0, "robot_red": 0, "robot_blue": 0},
        "2026-frc-robots": {"robot": 0},
        "autoscouting-wide-view": {"blue_robot": 0, "red_robot": 0},
        "2026-frc-robots": {"robot": 0},
        "pseudo_robots": {"robot": 0},
    },
    # densely labelled broadcast sets only: the airborne-only set (frc-fuel-2026-8qtjy) and the
    # partially labelled pile set (frc-fuel-zi0wy) would teach the model to ignore ground fuel
    "fuel": {
        "frc-fuel-2026-cvfs6": {"Fuel": 0, "fuel": 0},
        "frc2026_fuel": {"fuel": 0},
        "pseudo_fuel": {"fuel": 0},
    },
    # boxes around the visible bumpers only (PLAN.md 11). Checked by eye: these four box the
    # bumper ring / visible bumper segments; frc-2023-hwue2 v4 and robot-seg-06wv0 box whole
    # robots (box aspect ~1.2-1.5 vs ~2.3 for bumpers), so they are not used here
    "bumpers": {
        "ventura-2025": {"red": 0, "blue": 1},
        "frc-bumpers-h0p1g": {"red_bumper": 0, "blue_bumper": 1},
        "seg-rob-2026": {"red robot": 0, "blue robot": 1},  # bumper polygons -> boxes
        "bumper-based-detection-om4u6": {"robot_red": 0, "robot_blue": 1},
        # 2026 webcast frames: bumpers cut out of the robot pseudo-labels (scripts/pseudo_bumpers.py)
        "pseudo_bumpers": {"red_bumper": 0, "blue_bumper": 1},
    },
}
NAMES = {"robots": ["robot"], "fuel": ["fuel"], "bumpers": ["red_bumper", "blue_bumper"]}
# Roboflow exports preprocessed with "Stretch to" a square: broadcast frames squeezed from 16:9.
# Restored to 16:9 (labels are normalised, so they stay valid); thin bumpers depend on aspect.
UNSTRETCH = {"frc-bumpers-h0p1g", "seg-rob-2026"}
# un-augmented exports whose original names are bare frame numbers ("0019.jpg", "frame_29.jpg")
# from many videos: same stem != same frame, so every exported file is kept
FULL_STEM = {"ventura-2025", "frc-bumpers-h0p1g"}


def stem_key(p: Path) -> str:
    s = p.stem
    s = re.split(r"_(jpg|jpeg|png)\.rf\.", s)[0]
    s = re.sub(r"\.rf\.[0-9a-f]+$", "", s)
    return s.lower()


def to_box(vals: list[float]) -> tuple[float, float, float, float] | None:
    if len(vals) == 4:
        return tuple(vals)
    if len(vals) >= 6 and len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)
    return None


def content_box(img) -> tuple[int, int, int, int]:
    """Bounding box of the non-letterbox content (rows/cols that are not flat black/grey)."""
    import numpy as np
    g = img.mean(axis=2)
    rows = np.where(g.std(axis=1) > 6)[0]
    cols = np.where(g.std(axis=0) > 6)[0]
    if len(rows) < 10 or len(cols) < 10:
        return 0, 0, img.shape[1], img.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def mask_unlabeled_end_panes(img_path: Path, lines: list[str]):
    """FIRST-template frames: grey out the bottom camera panes if none of their robots are labeled.

    Returns a masked image array, or None when nothing needs masking.
    """
    import cv2
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from autoscout.views import detect_production

    img = cv2.imread(str(img_path))
    if img is None:
        return None
    x0, y0, x1, y1 = content_box(img)
    cw, ch = x1 - x0, y1 - y0
    if cw < 100 or abs(cw / max(ch, 1) - 16 / 9) > 0.12:
        return None
    content = cv2.resize(img[y0:y1, x0:x1], (1920, 1080))
    if detect_production(content) != "first":
        return None
    split = y0 + ch * 704 / 1080
    H, Wd = img.shape[:2]
    for ln in lines:
        _, cx, cy, bw, bh = map(float, ln.split())
        if cy * H > split:
            return None  # bottom panes are labeled in this image: keep as is
    out = img.copy()
    out[int(split):y1, x0:x1] = 114
    return out


def unstretch(img_path: Path):
    """A square (stretched) export resized back to 16:9, or None if the image is not square."""
    import cv2
    img = cv2.imread(str(img_path))
    if img is None or img.shape[0] != img.shape[1]:
        return None
    h = img.shape[0]
    return cv2.resize(img, (int(round(h * 16 / 9)), h), interpolation=cv2.INTER_CUBIC)


def link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main(kind: str, val_frac: float = 0.1, name: str | None = None):
    out = DS / (name or f"{kind}_v1")
    if out.exists():
        shutil.rmtree(out)
    for sp in ("train", "val"):
        (out / sp / "images").mkdir(parents=True)
        (out / sp / "labels").mkdir(parents=True)
    seen = set()
    stats = {}
    for src_name, cmap in RECIPES[kind].items():
        src = DS / src_name
        if not src.exists():
            print("skip (missing)", src_name)
            continue
        names = cmap
        y = src / "data.yaml"
        src_names = None
        if y.exists():
            d = yaml.safe_load(y.read_text())
            src_names = d.get("names")
            if isinstance(src_names, dict):
                src_names = [src_names[k] for k in sorted(src_names)]
        n_img = n_box = n_masked = 0
        for img in list(src.rglob("*.jpg")) + list(src.rglob("*.png")):
            if "images" not in img.parts:
                continue
            key = img.stem.lower() if src_name in FULL_STEM else stem_key(img)
            if key in seen:
                continue
            lab = Path(str(img).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")).with_suffix(".txt")
            lines = []
            if lab.exists():
                for ln in lab.read_text().splitlines():
                    v = ln.split()
                    if not v:
                        continue
                    cname = src_names[int(v[0])] if src_names else "robot"
                    if cname not in names:
                        continue
                    b = to_box([float(x) for x in v[1:]])
                    if b is None or b[2] <= 0 or b[3] <= 0:
                        continue
                    lines.append(f"{names[cname]} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}")
            if not lines and "pseudo" not in src_name:
                continue  # unlabeled frame in a source set: unknown, skip
            seen.add(key)
            h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            sp = "val" if h < val_frac else "train"
            uid = hashlib.md5(str(img).encode()).hexdigest()[:10]
            name = f"{src_name[:12]}_{key[:40]}_{uid}{img.suffix}"
            masked = mask_unlabeled_end_panes(img, lines) if kind in ("robots", "bumpers") and "pseudo" not in src_name else None
            if masked is None and src_name in UNSTRETCH and kind == "bumpers":
                masked = unstretch(img)
            if masked is not None:
                import cv2
                cv2.imwrite(str(out / sp / "images" / name), masked)
                n_masked += 1
            else:
                link(img, out / sp / "images" / name)
            (out / sp / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(lines))
            n_img += 1
            n_box += len(lines)
        stats[src_name] = (n_img, n_box)
        print(f"{src_name}: {n_img} images, {n_box} boxes, {n_masked} rewritten (end panes masked or unstretched)", flush=True)
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out).replace("\\", "/"), "train": "train/images", "val": "val/images",
        "names": {i: n for i, n in enumerate(NAMES[kind])}}))
    print("total", sum(s[0] for s in stats.values()), "images")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=sorted(RECIPES))
    ap.add_argument("--name", default=None, help="output dataset dir name (default <kind>_v1)")
    a = ap.parse_args()
    main(a.kind, name=a.name)
