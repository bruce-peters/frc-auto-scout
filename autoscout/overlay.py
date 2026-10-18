"""Read the broadcast score overlay: match clock, shift, live alliance fuel counters.

Two layouts are supported:

* ``first``  — FIRST's 2026 "FIRST AGE" audience overlay used at district / regional /
  championship events (counter boxes "N / 100" in the top corners, clock in the centre,
  "k / 6 :ss" shift box under the clock, yellow arrows next to the active hubs).
* ``chezy`` — the Chezy Champs overlay (stacked "N / 360" counters next to each score,
  clock under the centre logo).

ROIs are stored in 1920x1080 pixel coordinates and scaled to the actual frame size.
Readings are noisy, so `build_timeline` fuses them with the structure of a match: the clock
ticks once per second, counters never decrease.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AUTO_LEN, TELEOP_LEN

_READER = None


def reader():
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(["en"], gpu=True, verbose=False)
    return _READER


@dataclass
class Layout:
    name: str
    rois: dict[str, tuple[int, int, int, int]]  # x0, y0, x1, y1 at 1920x1080
    # text polarity per roi: True = light text on dark/coloured background
    light_text: dict[str, bool] = field(default_factory=dict)


FIRST = Layout(
    "first",
    rois={
        "blue_cnt": (88, 70, 226, 110),  # inside the box: its edges read as a stray "1"
        "red_cnt": (1742, 70, 1880, 110),
        "timer": (902, 64, 1018, 128),
        "shift": (902, 130, 1018, 165),
        "blue_score": (782, 62, 898, 128),
        "red_score": (1022, 62, 1138, 128),
        "blue_arrow": (302, 66, 348, 113),
        "red_arrow": (1572, 66, 1618, 113),
        "blue_teams": (450, 75, 780, 120),
        "red_teams": (1140, 75, 1470, 120),
    },
    light_text={"blue_cnt": True, "red_cnt": True, "blue_score": True, "red_score": True,
                "timer": False, "shift": False, "blue_teams": True, "red_teams": True},
)

CHEZY = Layout(
    "chezy",
    rois={
        # quals: "count" stacked over "360"; playoffs: one centred total -> take the top line
        "red_cnt": (555, 58, 640, 142),
        "blue_cnt": (1285, 58, 1370, 142),
        "timer": (912, 116, 1008, 152),
        "red_score": (735, 62, 880, 140),
        "blue_score": (1040, 62, 1185, 140),
        "red_teams": (495, 55, 555, 148),
        "blue_teams": (1362, 55, 1425, 148),
    },
    light_text={"red_cnt": True, "blue_cnt": True, "timer": False, "red_score": True,
                "blue_score": True, "red_teams": False, "blue_teams": False},
)

LAYOUTS = {"first": FIRST, "chezy": CHEZY}


def _crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    sx, sy = w / 1920.0, h / 1080.0
    x0, y0, x1, y1 = box
    return frame[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]


def _prep(crop: np.ndarray, light_text: bool, scale: float = 3.0) -> np.ndarray:
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if light_text and crop.ndim == 3:
        # white glyphs: min channel is high only for white/grey pixels, low for saturated bg
        g = crop.min(axis=2)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if light_text:
        b = 255 - b  # dark text on white for the recogniser
    return cv2.copyMakeBorder(b, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def ocr_text(img: np.ndarray, allow: str = "0123456789/:") -> tuple[str, float]:
    res = reader().readtext(img, allowlist=allow, detail=1, paragraph=False)
    if not res:
        return "", 0.0
    res.sort(key=lambda r: r[0][0][0])  # left to right
    txt = "".join(r[1] for r in res)
    conf = float(np.mean([r[2] for r in res]))
    return txt, conf


def ocr_top_line(img: np.ndarray, allow: str = "0123456789") -> tuple[str, float]:
    """Text of the top-most line only (stacked "count over threshold" counters)."""
    res = reader().readtext(img, allowlist=allow, detail=1, paragraph=False)
    if not res:
        return "", 0.0
    res.sort(key=lambda r: np.mean([p[1] for p in r[0]]))
    top_y = np.mean([p[1] for p in res[0][0]])
    line = [r for r in res if abs(np.mean([p[1] for p in r[0]]) - top_y) < 15]
    line.sort(key=lambda r: r[0][0][0])
    return "".join(r[1] for r in line), float(np.mean([r[2] for r in line]))


def detect_layout(frame: np.ndarray) -> str | None:
    """Guess which overlay (if any) is on this frame."""
    t = _crop(frame, FIRST.rois["timer"])
    b = _crop(frame, FIRST.rois["blue_score"]).reshape(-1, 3).mean(0)
    r = _crop(frame, FIRST.rois["red_score"]).reshape(-1, 3).mean(0)
    # FIRST: white timer box between a blue and a red score box
    if t.mean() > 170 and b[0] > b[2] + 40 and r[2] > r[0] + 40:
        return "first"
    rb = _crop(frame, CHEZY.rois["red_score"]).reshape(-1, 3).mean(0)
    bb = _crop(frame, CHEZY.rois["blue_score"]).reshape(-1, 3).mean(0)
    if rb[2] > rb[0] + 60 and bb[0] > bb[2] + 60:
        return "chezy"
    return None


def _arrow_on(crop: np.ndarray) -> bool:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    yellow = (hsv[..., 0] > 20) & (hsv[..., 0] < 40) & (hsv[..., 1] > 120) & (hsv[..., 2] > 150)
    return yellow.mean() > 0.35


READ_KEYS = ("blue_cnt", "red_cnt", "timer", "blue_arrow", "red_arrow")


def read_frame(frame: np.ndarray, layout: Layout, keys=READ_KEYS) -> dict:
    out: dict = {}
    for key, box in layout.rois.items():
        if keys and key not in keys:
            continue
        crop = _crop(frame, box)
        if key.endswith("_arrow"):
            out[key] = _arrow_on(crop)
            continue
        if key.endswith("_teams"):
            continue  # teams come from TBA
        img = _prep(crop, layout.light_text.get(key, True))
        if layout.name == "chezy" and key.endswith("_cnt"):
            txt, conf = ocr_top_line(img)
        else:
            txt, conf = ocr_text(img)
        out[key] = txt
        out[key + "_conf"] = conf
    return out


def parse_count(txt: str) -> int | None:
    m = re.match(r"^\s*(\d{1,3})\s*/", txt or "")
    if m:
        return int(m.group(1))
    # slash misread as a digit ("1571360" = "157/360"): strip a known RP threshold + separator
    t = (txt or "").strip()
    for thr in ("100", "240", "360", "500"):
        if len(t) >= 5 and t.endswith(thr) and t[-4] in "17l|":
            t2 = t[:-4]
            if 1 <= len(t2) <= 3 and t2.isdigit():
                return int(t2)
    m = re.match(r"^\s*(\d{1,3})\s*$", txt or "")
    return int(m.group(1)) if m else None


def parse_clock(txt: str) -> int | None:
    m = re.match(r"^\s*(\d):(\d\d)\s*$", txt or "")
    if not m:
        # the colon is sometimes dropped ("007" for 0:07); temporal voting rejects mistakes
        m = re.match(r"^\s*([0-2])([0-5]\d)\s*$", txt or "")
    if not m:
        return None
    s = int(m.group(1)) * 60 + int(m.group(2))
    return s if s <= TELEOP_LEN else None


def parse_shift(txt: str) -> tuple[int, int] | None:
    m = re.match(r"^\s*(\d)/6:?(\d\d)", (txt or "").replace(" ", ""))
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def scan(video, layout_name: str | None = None, hz: float = 4.0, t0: float = 0.0,
         t1: float | None = None, progress: bool = False) -> tuple[str | None, list[dict]]:
    """Read the overlay at `hz` samples per second. Returns (layout, rows)."""
    t1 = video.duration if t1 is None else t1
    if layout_name is None:
        votes = {}
        for t in np.linspace(t0 + 5, t1 - 5, 30):
            f = video.frame_at(float(t))
            if f is not None:
                k = detect_layout(f)
                if k is not None:  # long uploads are mostly non-match footage: ignore "none"
                    votes[k] = votes.get(k, 0) + 1
        layout_name = max(votes, key=votes.get) if votes and max(votes.values()) >= 3 else None
        if layout_name is None:
            return None, []
    layout = LAYOUTS[layout_name]
    step = max(1, int(round(video.fps / hz)))
    rows = []
    start = int(t0 * video.fps)
    end = int(t1 * video.fps)
    for idx, frame in video.iter_frames(start, end, step):
        if detect_layout(frame) != layout_name:
            rows.append({"t": idx / video.fps, "present": False})
            continue
        row = read_frame(frame, layout)
        row["t"] = idx / video.fps
        row["present"] = True
        rows.append(row)
        if progress and len(rows) % 100 == 0:
            print(f"  overlay t={row['t']:.1f}s")
    return layout_name, rows


@dataclass
class Timeline:
    """Mapping between video time and match time plus the alliance counter series."""

    auto_start: float | None  # video time of match start (t=0 match)
    teleop_start: float | None  # video time teleop clock starts (match t=20)
    counts: dict[str, list[tuple[float, int]]]  # alliance -> [(video_t, count)]
    active: dict[str, list[tuple[float, bool]]]  # alliance -> [(video_t, hub active?)]
    layout: str | None

    def match_time(self, vt: float) -> float | None:
        if self.auto_start is None:
            return None
        if self.teleop_start is not None and vt >= self.teleop_start:
            return AUTO_LEN + (vt - self.teleop_start)
        return vt - self.auto_start

    def video_time(self, mt: float) -> float | None:
        if self.auto_start is None:
            return None
        if mt >= AUTO_LEN and self.teleop_start is not None:
            return self.teleop_start + (mt - AUTO_LEN)
        return self.auto_start + mt

    def final_count(self, alliance: str) -> int | None:
        c = self.counts.get(alliance) or []
        return c[-1][1] if c else None

    def count_at(self, alliance: str, vt: float) -> int:
        c = self.counts.get(alliance) or []
        val = 0
        for t, n in c:
            if t > vt:
                break
            val = n
        return val


def _robust_offset(samples: list[tuple[float, int]], length: float) -> float | None:
    """Estimate the video time at which a countdown of `length` seconds started.

    The display shows ceil(remaining), so for a reading c at video time v the start s
    satisfies  v - (length - c) - 1 < s <= v - (length - c). Vote over a fine grid.
    """
    if len(samples) < 3:
        return None
    offs = np.array([v - (length - c) for v, c in samples])
    grid = np.arange(offs.min() - 1.5, offs.max() + 0.5, 0.02)
    votes = np.array([np.sum((offs - 1 < g) & (g <= offs)) for g in grid])
    best = votes.max()
    if best < max(3, 0.4 * len(samples)):
        return None
    return float(grid[votes == best].mean())


def _clean_counts(raw: list[tuple[float, int]], max_rate: float = 30.0) -> list[tuple[float, int]]:
    """Keep a non-decreasing, physically plausible subsequence of counter readings.

    Longest chain where count never decreases and never rises faster than `max_rate`/s
    (plus slack), solved by O(n^2) DP — n is a few hundred.
    """
    n = len(raw)
    if n == 0:
        return []
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        ti, ci = raw[i]
        for j in range(max(0, i - 60), i):
            tj, cj = raw[j]
            if cj <= ci <= cj + max_rate * (ti - tj) + 3 and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                prev[i] = j
    i = int(np.argmax(best))
    chain = []
    while i >= 0:
        chain.append(raw[i])
        i = prev[i]
    chain.reverse()
    # require each value to be seen twice in a row (or be the last) to suppress one-offs
    out = []
    for k, (t, c) in enumerate(chain):
        if out and c == out[-1][1]:
            continue
        nxt = chain[k + 1][1] if k + 1 < len(chain) else c
        if nxt >= c:
            out.append((t, c))
    return out


def build_timeline(layout: str | None, rows: list[dict]) -> Timeline:
    if not rows or layout is None:
        return Timeline(None, None, {}, {}, layout)
    clocks = [(r["t"], parse_clock(r.get("timer", ""))) for r in rows if r.get("present")]
    clocks = [(t, c) for t, c in clocks if c is not None]
    # The clock reads 0:20 -> 0:00 in auto and 2:20 -> 0:00 in teleop. Split on the first
    # reading > 20 that follows auto readings.
    auto = [(t, c) for t, c in clocks if c <= 20]
    tele = [(t, c) for t, c in clocks if c > 20]
    teleop_start = _robust_offset([(t, c) for t, c in tele], TELEOP_LEN)
    if teleop_start is not None:
        auto = [(t, c) for t, c in auto if t < teleop_start - 1]
        tele_low = [(t, c) for t, c in clocks if c <= 20 and t > teleop_start + 100]
        teleop_start2 = _robust_offset(tele + tele_low, TELEOP_LEN)
        if teleop_start2 is not None:
            teleop_start = teleop_start2
    # Pre-match "0:20" readings (before start) would bias the auto fit: drop leading 0:20s
    auto = [(t, c) for t, c in auto if c < 20] or auto
    auto_start = _robust_offset(auto, AUTO_LEN)
    # Only this match's readings: a video that also shows the previous or next match (2026cc_qm14 runs on into
    # qm15) would otherwise have its counter series picked from the other match by _clean_counts.
    lo, hi = -1e9, 1e9
    if auto_start is not None:
        lo = auto_start - 10.0
    if teleop_start is not None:
        hi = teleop_start + TELEOP_LEN + 20.0
    counts, active = {}, {}
    for al in ("red", "blue"):
        raw = [(r["t"], parse_count(r.get(f"{al}_cnt", ""))) for r in rows if r.get("present") and lo <= r["t"] <= hi]
        raw = [(t, c) for t, c in raw if c is not None]
        counts[al] = _clean_counts(raw)
        active[al] = [(r["t"], bool(r.get(f"{al}_arrow"))) for r in rows
                      if r.get("present") and f"{al}_arrow" in r]
    return Timeline(auto_start, teleop_start, counts, active, layout)
