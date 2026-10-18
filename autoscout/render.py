"""Annotated match videos: what the system sees, frame by frame.

The output stacks the original webcast (scaled) over an info panel:

* robot boxes labelled with the team number the tracker assigned (one colour per team),
* launch arcs it detected, drawn in the shooter's colour,
* the two score-overlay counters it reads with OCR, boxed, with the value it read,
* a top-down field with every robot's tracked position and a short trail,
* a running per-team fuel tally (the alliance counter, split among its robots).

    python -m autoscout.render 2026casnv_qm2              # whole match
    python -m autoscout.render 2026casnv_qm2 --t0 60 --t1 95
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import config, overlay, views
from . import field as fieldmod
from .video import Video, video_path

# BGR; reds for the red alliance, blues for blue
RED_COLS = [(40, 40, 230), (0, 150, 255), (180, 80, 255)]
BLUE_COLS = [(255, 140, 30), (230, 220, 0), (255, 80, 160)]


def _ffmpeg_writer(path: Path, w: int, h: int, fps: float):
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = "ffmpeg"
    cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
           "-preset", "veryfast", "-movflags", "+faststart", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def _label(img, text, org, color, scale=0.9, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 2, y - th - 6), (x + tw + 4, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _field_panel(w: int, h: int):
    """Top-down field image and a metres->pixels mapper."""
    L, Wd = config.FIELD_LENGTH, config.FIELD_WIDTH
    pad = 12
    s = min((w - 2 * pad) / L, (h - 2 * pad) / Wd)
    ox, oy = (w - L * s) / 2, (h - Wd * s) / 2
    img = np.full((h, w, 3), 30, np.uint8)

    def m2p(x, y):
        return int(ox + x * s), int(oy + (Wd - y) * s)  # +y up on screen

    cv2.rectangle(img, m2p(0, 0), m2p(L, Wd), (70, 70, 70), -1)
    cv2.rectangle(img, m2p(0, 0), m2p(L, Wd), (220, 220, 220), 2)
    cv2.line(img, m2p(L / 2, 0), m2p(L / 2, Wd), (230, 230, 230), 1)
    z = fieldmod.ALLIANCE_ZONE_DEPTH
    cv2.line(img, m2p(z, 0), m2p(z, Wd), (200, 120, 40), 1)
    cv2.line(img, m2p(L - z, 0), m2p(L - z, Wd), (60, 60, 200), 1)
    for al, col in (("blue", (200, 120, 40)), ("red", (60, 60, 200))):
        cx, cy = fieldmod.hub_center(al)
        hh = fieldmod.HUB_HALF
        cv2.rectangle(img, m2p(cx - hh, cy + hh), m2p(cx + hh, cy - hh), col, -1)
    return img, m2p, s


def render(match_key: str, t0: float | None = None, t1: float | None = None, out: Path | None = None,
           fps_out: float = 30.0, scale: float = 0.5) -> Path:
    od = config.OUTPUT / match_key
    res = json.loads((od / "result.json").read_text())
    dets = json.loads((od / "dets.json").read_text())
    tl = res["timeline"]
    a0, ts = tl.get("auto_start") or 6.13, tl.get("teleop_start")

    def to_mt(t):
        return (20 + t - ts) if ts and t >= ts else t - a0

    def to_vt(mt):
        return (ts + mt - 20) if ts and mt >= 20 else a0 + mt

    t0 = a0 - 2 if t0 is None else t0
    t1 = (ts or a0 + 25) + 143 if t1 is None else t1
    teams = {"red": [], "blue": []}
    for t, v in res["teams"].items():
        if not str(t).startswith("_"):
            teams[v["alliance"]].append(int(t))
    color = {}
    for al, cols in (("red", RED_COLS), ("blue", BLUE_COLS)):
        for i, t in enumerate(sorted(teams[al])):
            color[t] = cols[i % 3]
    scale_al = {al: res["teams"][f"_{al}"].get("scale", 1.0) for al in ("red", "blue")}
    alliance_of = {t: al for al in teams for t in teams[al]}

    v = Video(video_path(match_key))
    prod = views.detect_production(v.frame_at((t0 + t1) / 2))
    offs = {vw.name: vw.offset(v.width, v.height) for vw in views.PRODUCTIONS[prod]}
    lay = overlay.LAYOUTS.get(tl.get("layout") or "", None)

    by_t: dict[float, list] = {}
    for row in dets:
        by_t.setdefault(row[0], []).append(row)
    dkeys = np.array(sorted(by_t)) if by_t else np.zeros(0)
    paths = {int(k): np.array([p for p in pv if p[0] is not None]) for k, pv in res["paths"].items() if pv}
    log = np.array([(m, tm, n) for m, tm, n in res.get("fuel_log", [])]) if res.get("fuel_log") else np.zeros((0, 3))
    counts = {al: tl.get("counts", {}).get(al) or [] for al in ("red", "blue")}

    W, H = int(v.width * scale), int(v.height * scale)
    PH = int(H * 0.52)
    out = out or od / (f"demo_{match_key}.mp4" if (t0 <= a0 and t1 >= (ts or 0) + 140) else
                       f"demo_{match_key}_{t0:.0f}_{t1:.0f}.mp4")
    wr = _ffmpeg_writer(out, W, H + PH, fps_out)
    fw = int(W * 0.62)
    field_base, m2p, mpx = _field_panel(fw, PH)
    step = max(1, int(round(v.fps / fps_out)))
    for idx, frame in v.iter_frames(int(t0 * v.fps), int(t1 * v.fps), step):
        t = idx / v.fps
        mt = to_mt(t)
        img = frame
        # ---- robots
        if len(dkeys):
            j = int(np.argmin(np.abs(dkeys - t)))
            if abs(dkeys[j] - t) < 0.12:
                for row in by_t[dkeys[j]]:
                    _, view, x0, y0, x1, y1, al, team, sid = row[:9]
                    c = color.get(team, (190, 190, 190))
                    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), c, 4)
                    _label(img, str(team) if team else "?", (x0, y0 - 6), c, 0.9, 2)
        # ---- launch arcs (last 0.8 s)
        for sh in res["shots"]:
            if sh["t"] <= t <= sh["t"] + 0.8:
                ox, oy = offs.get(sh["view"], (0, 0))
                pts = np.array([[p[1] + ox, p[2] + oy] for p in sh["pts"] if p[0] <= t], np.int32)
                if len(pts) >= 2:
                    cv2.polylines(img, [pts], False, color.get(sh["team"], (255, 255, 255)), 4, cv2.LINE_AA)
        # ---- overlay counters the OCR reads
        if lay is not None:
            for al in ("red", "blue"):
                box = lay.rois.get(f"{al}_cnt")
                if box is None:
                    continue
                sx, sy = v.width / 1920, v.height / 1080
                x0, y0, x1, y1 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
                cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 3, y1 + 3), (0, 255, 255), 3)
                val = 0
                for tt, c in counts[al]:
                    if tt > t:
                        break
                    val = c
                _label(img, f"OCR {val}", (x0, y1 + 34), (0, 255, 255), 0.9, 2)
        top = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        # ---- panel: field map
        panel = np.full((PH, W, 3), 22, np.uint8)
        fimg = field_base.copy()
        for team, arr in paths.items():
            if not len(arr):
                continue
            k = np.searchsorted(arr[:, 0], mt)
            near = [arr[i] for i in (k - 1, k) if 0 <= i < len(arr) and abs(arr[i][0] - mt) < 0.6]
            trail = arr[(arr[:, 0] <= mt) & (arr[:, 0] >= mt - 3)]
            c = color.get(team, (200, 200, 200))
            if len(trail) >= 2:
                cv2.polylines(fimg, [np.array([m2p(p[1], p[2]) for p in trail], np.int32)], False, c, 2, cv2.LINE_AA)
            if near:
                p = min(near, key=lambda q: abs(q[0] - mt))
                cx, cy = m2p(p[1], p[2])
                cv2.circle(fimg, (cx, cy), max(5, int(0.45 * mpx)), c, -1)
                cv2.putText(fimg, str(team), (cx + 7, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                            cv2.LINE_AA)
        panel[:, :fw] = fimg
        # ---- panel: running fuel per team
        x = fw + 14
        cv2.putText(panel, "FUEL scored (video split)", (x, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1,
                    cv2.LINE_AA)
        y = 52
        maxv = max([1.0] + [res["teams"][str(tm)]["fuel"] for tm in color])
        for al in ("red", "blue"):
            for team in sorted(teams[al]):
                got = float(log[(log[:, 1] == team) & (log[:, 0] <= mt), 2].sum()) * scale_al[al] if len(log) else 0.0
                c = color[team]
                cv2.putText(panel, f"{team}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
                bw = int((W - x - 80) * got / maxv)
                cv2.rectangle(panel, (x + 62, y - 14), (x + 62 + bw, y + 2), c, -1)
                cv2.putText(panel, f"{got:.0f}", (x + 68 + bw, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1,
                            cv2.LINE_AA)
                y += 30
            y += 8
        per = "auto" if mt < 20 else ("teleop" if mt <= 160 else "post")
        cv2.putText(panel, f"match {max(0.0, mt):5.1f}s ({per})", (x, PH - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)
        wr.stdin.write(np.vstack([top, panel]).tobytes())
    wr.stdin.close()
    wr.wait()
    return out


def contact_sheet(clip: Path, n: int = 6, out: Path | None = None) -> Path:
    cap = cv2.VideoCapture(str(clip))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ims = []
    for i in np.linspace(0, N - 1, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            ims.append(f)
    rows = [np.hstack(ims[i:i + 2]) for i in range(0, len(ims) - 1, 2)]
    out = out or clip.with_suffix(".jpg")
    cv2.imwrite(str(out), np.vstack(rows))
    return out


def main():
    ap = argparse.ArgumentParser(prog="python -m autoscout.render")
    ap.add_argument("match")
    ap.add_argument("--t0", type=float, help="video seconds (default: match start)")
    ap.add_argument("--t1", type=float, help="video seconds (default: match end)")
    ap.add_argument("--scale", type=float, default=0.5)
    a = ap.parse_args()
    print(render(a.match, a.t0, a.t1, scale=a.scale))


if __name__ == "__main__":
    main()
