"""Training set for the flying-fuel detector (one class, "ball" = FUEL in the air).

1. Roboflow frc-fuel-2026-8qtjy (labels only airborne fuel). Its frames are full 16:9 webcast
   frames stretched to 640x640; they are un-stretched back to 1138x640 (labels are normalised,
   so they stay valid).
2. Pseudo-labels from our own webcasts: blobs on flight tracks that pass `fuel.is_shot`
   (box = the motion blob's box). Every other moving-yellow pixel (looser thresholds than the
   tracker's) is greyed out, so balls in the air that were not confidently tracked are not
   taught as background. Static fuel on the floor stays as negative examples.
   Frames are cut into 960x540 tiles at native resolution (inference runs on full 1080p
   frames at imgsz 1920, so the ball scale matches).

Matches of the held-out events (default 2026joh, 2026cc) are never used.
"""
from __future__ import annotations

import argparse
import pickle
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, fuel, pipeline, views  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402

TILE_W, TILE_H = 960, 540
RF = config.DATA / "datasets" / "frc-fuel-2026-8qtjy"


def loose_moving_yellow(cur: np.ndarray, prev_g: np.ndarray, cur_g: np.ndarray, next_g: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yel = (h >= fuel.H_LO - 3) & (h <= fuel.H_HI + 4) & (s >= 50) & (v >= 70)
    mov = (cv2.absdiff(cur_g, prev_g) > 12) | (cv2.absdiff(cur_g, next_g) > 12)
    return (yel & mov).astype(np.uint8)


def shot_points(md, fps: float) -> dict[int, list]:
    """source frame index -> [(view, u, v)] for every point of every is_shot track."""
    out = defaultdict(list)
    for vn, trs in md.tracks_all.items():
        cam_track = md.cams.get(vn)
        for tr in trs:
            cam = cam_track.at(tr.pts[0][0]) if cam_track is not None else None
            if not fuel.is_shot(tr, pipeline._px_per_m(cam, tr.pts[0][1], tr.pts[0][2])):
                continue
            for t, u, v, _ in tr.pts:
                out[int(round(t * fps))].append((vn, u, v))
    return out


def label_frame(video: Video, idx: int, pts: list, vlist: list, k: int = 2):
    """-> (image with uncertain moving yellow greyed out, [x0, y0, x1, y1] labels, n matched, n points)."""
    frames = [video.frame(i) for i in range(idx - k, idx + k + 1)]
    if any(f is None for f in frames):
        return None
    prev, cur, nxt = frames[0], frames[k], frames[-1]
    pg, cg, ng = (cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in (prev, cur, nxt))
    cy = fuel.yellow_mask(cur)
    W, H = cur.shape[1], cur.shape[0]
    boxes, matched = [], 0
    for v in vlist:
        vp = [(u, vv) for vn, u, vv in pts if vn == v.name]
        if not vp:
            continue
        x0, y0, x1, y1 = v.scaled(W, H)
        valid = np.ones((y1 - y0, x1 - x0), np.uint8)
        for mx0, my0, mx1, my1 in v.mask_boxes(W, H):
            valid[max(0, my0):max(0, my1), max(0, mx0):max(0, mx1)] = 0
        blobs = fuel.view_blobs(cg[y0:y1, x0:x1], pg[y0:y1, x0:x1], ng[y0:y1, x0:x1], cy[y0:y1, x0:x1], valid)
        if not blobs:
            continue
        B = np.array([(b.u, b.v) for b in blobs])
        used = set()
        for u, vv in vp:
            d = np.hypot(B[:, 0] - u, B[:, 1] - vv)
            j = int(np.argmin(d))
            if d[j] > 1.0 or j in used:
                continue
            used.add(j)
            matched += 1
            b = blobs[j]
            # the blob was dilated by 1 px on each side; keep boxes >= 5 px
            bx0, by0 = x0 + b.x0 + 1, y0 + b.y0 + 1
            bw, bh = max(5.0, b.w - 2), max(5.0, b.h - 2)
            cx, cyy = bx0 + (b.w - 2) / 2, by0 + (b.h - 2) / 2
            boxes.append([cx - bw / 2, cyy - bh / 2, cx + bw / 2, cyy + bh / 2])
    if not boxes:
        return None, [], matched, len(pts)
    # grey out moving yellow that is not a labelled ball
    loose = loose_moving_yellow(cur, pg, cg, ng)
    n, lab = cv2.connectedComponents(loose, connectivity=8)
    lab_core = np.zeros((H, W), bool)
    lab_pad = np.zeros((H, W), bool)
    lab_near = np.zeros((H, W), bool)
    for bx0, by0, bx1, by1 in boxes:
        s = max(bx1 - bx0, by1 - by0)
        lab_core[int(by0):int(np.ceil(by1)), int(bx0):int(np.ceil(bx1))] = True
        p = int(max(6, 0.6 * s))
        lab_pad[max(0, int(by0) - p):int(by1) + p, max(0, int(bx0) - p):int(bx1) + p] = True
        lab_near[max(0, int(by0) - 2):int(by1) + 2, max(0, int(bx0) - 2):int(bx1) + 2] = True
    touch = np.unique(lab[lab_core])
    touch = touch[touch > 0]
    grey = (lab > 0) & ~np.isin(lab, touch)
    grey |= np.isin(lab, touch) & ~lab_pad  # a stream merged into one component: keep only the labelled ball
    grey = cv2.dilate(grey.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    grey &= ~lab_near
    img = cur.copy()
    img[grey] = 114
    return img, boxes, matched, len(pts)


def write_tiles(img: np.ndarray, boxes: list, stem: str, out: Path, rng: random.Random, neg_frac: float) -> int:
    H, W = img.shape[:2]
    n = 0
    for ty in range(0, H - TILE_H + 1, TILE_H):
        for tx in range(0, W - TILE_W + 1, TILE_W):
            lines = []
            for x0, y0, x1, y1 in boxes:
                cx0, cy0 = max(x0, tx), max(y0, ty)
                cx1, cy1 = min(x1, tx + TILE_W), min(y1, ty + TILE_H)
                if cx1 <= cx0 or cy1 <= cy0:
                    continue
                if (cx1 - cx0) * (cy1 - cy0) < 0.5 * (x1 - x0) * (y1 - y0):
                    continue
                lines.append(f"0 {((cx0 + cx1) / 2 - tx) / TILE_W:.6f} {((cy0 + cy1) / 2 - ty) / TILE_H:.6f} "
                             f"{(cx1 - cx0) / TILE_W:.6f} {(cy1 - cy0) / TILE_H:.6f}")
            if not lines and rng.random() > neg_frac:
                continue
            name = f"{stem}_{tx}_{ty}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), img[ty:ty + TILE_H, tx:tx + TILE_W],
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            (out / "labels" / f"{name}.txt").write_text("\n".join(lines))
            n += 1
    return n


def copy_roboflow(split_src: str, dst: Path):
    for f in sorted((RF / split_src / "images").glob("*.jpg")):
        img = cv2.imread(str(f))
        img = cv2.resize(img, (1138, 640), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst / "images" / f"rf_{f.name}"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lab = RF / split_src / "labels" / (f.stem + ".txt")
        shutil.copy(lab, dst / "labels" / f"rf_{f.stem}.txt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(config.DATA / "datasets" / "flying_fuel"))
    ap.add_argument("--holdout", nargs="*", default=["2026joh", "2026cc"])
    ap.add_argument("--per-match", type=int, default=30)
    ap.add_argument("--spacing", type=float, default=0.75, help="s between sampled frames")
    ap.add_argument("--neg-frac", type=float, default=0.25, help="keep this share of label-free tiles")
    ap.add_argument("--val-every", type=int, default=8, help="every n-th match goes to val")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val", "test"):
        (out / s / "images").mkdir(parents=True)
        (out / s / "labels").mkdir(parents=True)
    copy_roboflow("train", out / "train")
    copy_roboflow("valid", out / "val")
    copy_roboflow("test", out / "test")
    (out / "data.yaml").write_text(f"path: {out.as_posix()}\ntrain: train/images\nval: val/images\n"
                                   "test: test/images\nnames:\n  0: ball\n")
    keys = sorted(p.parent.name for p in config.OUTPUT.glob("*/scan.pkl")
                  if not any(p.parent.name.startswith(h + "_") for h in a.holdout))
    rng = random.Random(0)
    sheet = []
    tot_m = tot_p = tot_tiles = 0
    for mi, key in enumerate(keys):
        vp = video_path(key)
        if not vp.exists():
            continue
        video = Video(vp)
        if video.width != 1920 or video.height != 1080:
            print("skip (not 1080p)", key)
            continue
        with open(config.OUTPUT / key / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        vlist = views.PRODUCTIONS[md.prod]
        pts = shot_points(md, video.fps)
        cand = sorted(pts, key=lambda i: -len(pts[i]))
        chosen = []
        gap = a.spacing * video.fps
        for i in cand:
            if len(chosen) >= a.per_match:
                break
            if all(abs(i - c) >= gap for c in chosen):
                chosen.append(i)
        split = "val" if mi % a.val_every == 0 else "train"
        m_match = m_pts = n_tiles = 0
        for i in sorted(chosen):
            r = label_frame(video, i, pts[i], vlist)
            if r is None:
                continue
            img, boxes, nm, npt = r
            m_match += nm
            m_pts += npt
            if img is None:
                continue
            n_tiles += write_tiles(img, boxes, f"{key}_{i}", out / split, rng, a.neg_frac)
            if len(sheet) < 6 and rng.random() < 0.05:
                vis = img.copy()
                for x0, y0, x1, y1 in boxes:
                    cv2.rectangle(vis, (int(x0) - 3, int(y0) - 3), (int(x1) + 3, int(y1) + 3), (255, 0, 255), 2)
                sheet.append(cv2.resize(vis, (960, 540)))
        video.close()
        tot_m += m_match
        tot_p += m_pts
        tot_tiles += n_tiles
        print(f"{key} [{split}] frames={len(chosen)} shot-points matched {m_match}/{m_pts} tiles={n_tiles}", flush=True)
    if len(sheet) >= 2:
        rows = [np.hstack(sheet[i:i + 2]) for i in range(0, len(sheet) - 1, 2)]
        (config.DATA / "debug").mkdir(exist_ok=True)
        cv2.imwrite(str(config.DATA / "debug" / "flying_fuel_sheet.jpg"), np.vstack(rows))
    print(f"total: matched {tot_m}/{tot_p} shot points, {tot_tiles} pseudo-label tiles")


if __name__ == "__main__":
    main()
