"""Bumper boxes for our own 2026 webcast frames, cut out of the whole-robot pseudo-labels.

The public bumper sets have only a few hundred REBUILT frames. data/datasets/pseudo_robots
(scripts/pseudo_label.py: confident robot boxes with a clear bumper colour, mid-confidence boxes
greyed out) covers every event's broadcast. Inside each robot box the bumper is the saturated
red/blue band at the bottom: colour mask (same thresholds as robots.bumper_alliance) in the
lower part of the box, closed horizontally over the white digits and FUEL in front, union of the components that
reach the lower half. Robots whose bumper cannot be cut out cleanly are greyed out (unknown),
so they are not taught as background.

Frames of --exclude events (default 2026cc, the bumper experiment's test event) are skipped.
Writes data/datasets/pseudo_bumpers/{images,labels} (classes 0 red_bumper, 1 blue_bumper) and a
contact sheet data/datasets/pseudo_bumpers_sheet.jpg.
"""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "datasets" / "pseudo_robots"
OUT = ROOT / "data" / "datasets" / "pseudo_bumpers"


def colour_masks(crop: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s_ok = (hsv[..., 1] > 110) & (hsv[..., 2] > 60)
    hue = hsv[..., 0]
    return {"red": s_ok & ((hue < 8) | (hue > 165)), "blue": s_ok & (hue > 95) & (hue < 130)}


def cut_bumper(img: np.ndarray, box: tuple[int, int, int, int]) -> tuple[tuple[int, int, int, int], str] | None:
    """(bumper box in image px, alliance) inside a robot box, or None if not clean."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    if w < 12 or h < 10:
        return None
    ex0, ex1 = max(0, x0 - int(0.03 * w)), min(W, x1 + int(0.03 * w))
    ey0, ey1 = max(0, y0 + int(0.35 * h)), min(H, y1 + int(0.08 * h) + 1)  # bumpers: lower part of the box
    crop = img[ey0:ey1, ex0:ex1]
    if crop.size == 0:
        return None
    masks = colour_masks(crop)
    al = max(masks, key=lambda k: masks[k].sum())
    m = masks[al].astype(np.uint8)
    other = masks["blue" if al == "red" else "red"].sum()
    if m.sum() < 12 or other > 0.35 * m.sum():
        return None
    # bridge the white digits and FUEL balls in front of the bumper (a ball is ~17 % of a robot)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 4) | 1, 3))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    areas = st[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i in range(n - 1) if areas[i] >= 0.15 * areas.max()
            and st[i + 1, cv2.CC_STAT_TOP] + st[i + 1, cv2.CC_STAT_HEIGHT] > 0.4 * m.shape[0]]
    if not keep:
        return None
    bx0 = min(st[i, cv2.CC_STAT_LEFT] for i in keep)
    by0 = min(st[i, cv2.CC_STAT_TOP] for i in keep)
    bx1 = max(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in keep)
    by1 = max(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in keep)
    bw, bh = bx1 - bx0, by1 - by0
    fill = np.isin(lab[by0:by1, bx0:bx1], keep).mean()
    if bw < 0.5 * w or bh < 3 or bw < 1.2 * bh or bh > 0.6 * h or fill < 0.3:
        return None
    if abs(ey0 + by1 - y1) > 0.2 * h:  # bumpers sit at the bottom of the robot box
        return None
    return (ex0 + bx0, ey0 + by0, ex0 + bx1, ey0 + by1), al


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=["2026cc"])
    a = ap.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    (OUT / "labels").mkdir(parents=True)
    n_img = n_ok = n_grey = 0
    sheet = []
    labs = sorted((SRC / "labels").glob("*.txt"))
    for lab in labs:
        if any(lab.stem.startswith(e + "_") for e in a.exclude):
            continue
        ip = SRC / "images" / (lab.stem + ".jpg")
        img = cv2.imread(str(ip))
        if img is None:
            continue
        H, W = img.shape[:2]
        lines, grey = [], []
        for ln in lab.read_text().splitlines():
            v = ln.split()
            if len(v) != 5:
                continue
            cx, cy, bw, bh = (float(x) for x in v[1:])
            box = (int((cx - bw / 2) * W), int((cy - bh / 2) * H), int((cx + bw / 2) * W), int((cy + bh / 2) * H))
            r = cut_bumper(img, box)
            if r is None:
                grey.append(box)
                continue
            (x0, y0, x1, y1), al = r
            lines.append(f"{0 if al == 'red' else 1} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} "
                         f"{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}")
            if len(sheet) < 400 and random.random() < 0.15:
                sheet.append((img, box, (x0, y0, x1, y1), al))
        out_img = OUT / "images" / ip.name
        if grey:
            img = img.copy()
            for x0, y0, x1, y1 in grey:
                img[max(0, y0):y1, max(0, x0):x1] = 114
            cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            try:
                os.link(ip, out_img)
            except OSError:
                shutil.copy2(ip, out_img)
        (OUT / "labels" / lab.name).write_text("\n".join(lines))
        n_img += 1
        n_ok += len(lines)
        n_grey += len(grey)
    (OUT / "data.yaml").write_text("names:\n- red_bumper\n- blue_bumper\nnc: 2\n")
    print(f"{n_img} images, {n_ok} bumper boxes, {n_grey} robots greyed out (no clean bumper)")
    # contact sheet: robot box (grey) and the cut bumper box (alliance colour), 48 random crops
    random.seed(0)
    tiles = []
    for img, rb, bb, al in random.sample(sheet, min(48, len(sheet))):
        x0, y0, x1, y1 = rb
        pad = max(8, (x1 - x0) // 3)
        c = img[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad].copy()
        ox, oy = max(0, x0 - pad), max(0, y0 - pad)
        cv2.rectangle(c, (x0 - ox, y0 - oy), (x1 - ox, y1 - oy), (200, 200, 200), 1)
        cv2.rectangle(c, (bb[0] - ox, bb[1] - oy), (bb[2] - ox, bb[3] - oy),
                      (0, 0, 255) if al == "red" else (255, 160, 0), 1)
        tiles.append(cv2.resize(c, (160, 160)))
    if tiles:
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT.parent / "pseudo_bumpers_sheet.jpg"),
                    np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))


if __name__ == "__main__":
    main()
