"""Pseudo-label our own 2026 webcast frames with the current robot detector.

Keeps boxes that are confident AND have a clear red/blue bumper, writes YOLO labels into
data/datasets/pseudo_robots/{images,labels}. Mid-confidence boxes are greyed out in the saved
image rather than left unlabeled, so the set has few false negatives.
With --panes the low end-camera views are also detected on 2x upscaled crops and merged with the
full-frame boxes: robots there are small, and the zoomed pass finds many that the full-frame pass
misses, so the retrained model learns to find them in full frames. End-view boxes use their own
(lower) threshold --hi_end: v1 learned end-camera robots from few labels, so it is under-confident
there (0.40-0.55 boxes with a clear bumper colour were ~95 % real robots on a 80-crop check).
Also writes a contact sheet for a quick visual check (plus an end-pane sheet with --panes).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import views  # noqa: E402
from autoscout.config import DATA  # noqa: E402
from autoscout.robots import bumper_alliance  # noqa: E402


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def detect(m, img, lo: float, panes: bool, scale: float = 2.0) -> list[tuple[tuple, float]]:
    """Full-frame boxes, optionally merged (greedy NMS, max conf) with upscaled end-view boxes."""
    r = m.predict(img, imgsz=1280, conf=lo, verbose=False)[0]
    boxes = [(tuple(map(float, b)), float(c)) for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    if not panes:
        return boxes
    H, W = img.shape[:2]
    for v in views.PRODUCTIONS[views.detect_production(img)]:
        if v.name == "main":
            continue
        x0, y0, x1, y1 = v.scaled(W, H)
        up = cv2.resize(img[y0:y1, x0:x1], None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        r = m.predict(up, imgsz=1280, conf=lo, verbose=False)[0]
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            bx = (b[0] / scale + x0, b[1] / scale + y0, b[2] / scale + x0, b[3] / scale + y0)
            # boxes cut by the crop border are partial robots: only the full-frame pass may label them
            if min(b[0], b[1], up.shape[1] - b[2], up.shape[0] - b[3]) < 3:
                continue
            boxes.append((tuple(map(float, bx)), float(c)))
    boxes.sort(key=lambda bc: -bc[1])
    keep: list[tuple[tuple, float]] = []
    for b, c in boxes:
        if all(_iou(b, k) < 0.5 for k, _ in keep):
            keep.append((b, c))
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pool", default=str(DATA / "frame_pool"))
    ap.add_argument("--out", default=str(DATA / "datasets" / "pseudo_robots"))
    ap.add_argument("--hi", type=float, default=0.55)
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--max", type=int, default=1200)
    ap.add_argument("--panes", action="store_true", help="also detect on 2x upscaled end-camera views")
    ap.add_argument("--hi_end", type=float, default=None, help="confidence to label a box in an end view (default --hi)")
    a = ap.parse_args()
    hi_end = a.hi if a.hi_end is None else a.hi_end
    from ultralytics import YOLO

    m = YOLO(a.model)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text("names:\n- robot\nnc: 1\n")
    files = sorted(Path(a.pool).glob("*.jpg"))
    random.Random(0).shuffle(files)
    kept = skipped = 0
    sheet, pane_sheet = [], []
    for f in files[: a.max]:
        img = cv2.imread(str(f))
        H, W = img.shape[:2]
        lines, ambiguous = [], []
        vis = img.copy()
        ends = [v.scaled(W, H) for v in views.PRODUCTIONS[views.detect_production(img)] if v.name != "main"]
        for b, c in detect(m, img, a.lo, a.panes):
            x0, y0, x1, y1 = b
            if y1 < 170 and x0 > 440 and x1 < 1480:
                continue  # inside the score bar
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            hi = hi_end if any(v[0] <= cx < v[2] and v[1] <= cy < v[3] for v in ends) else a.hi
            al, frac = bumper_alliance(img[int(y0):int(y1), int(x0):int(x1)])
            if c >= hi and al is not None and frac > 0.7:
                lines.append(f"0 {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 3)
            else:
                ambiguous.append((int(x0), int(y0), int(x1), int(y1)))
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 2)
        if len(lines) < 2:
            skipped += 1
            continue
        # unsure boxes are greyed out instead of left unlabeled (no false negatives)
        save = img.copy()
        for x0, y0, x1, y1 in ambiguous:
            save[max(0, y0):y1, max(0, x0):x1] = 114
        cv2.imwrite(str(out / "images" / f.name), save)
        (out / "labels" / (f.stem + ".txt")).write_text("\n".join(lines))
        kept += 1
        if len(sheet) < 6:
            sheet.append(cv2.resize(vis, (960, 540)))
        if a.panes and len(pane_sheet) < 16 and kept % 5 == 0:
            pane_sheet.append(cv2.resize(vis[int(H * 745 / 1080):, :], (1280, 240)))
    if sheet:
        rows = [np.hstack(sheet[i:i + 2]) for i in range(0, len(sheet) - 1, 2)]
        cv2.imwrite(str(DATA / "debug" / "pseudo_robots_sheet.jpg"), np.vstack(rows))
    if pane_sheet:
        cv2.imwrite(str(DATA / "debug" / "pseudo_robots_panes_sheet.jpg"), np.vstack(pane_sheet))
    print(f"kept {kept}, skipped {skipped}")


if __name__ == "__main__":
    main()
