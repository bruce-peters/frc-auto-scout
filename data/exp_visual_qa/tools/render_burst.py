"""Render annotated broadcast frames for scoring bursts (from find_bursts.py on the in-memory cache).

Per burst, N frames from ~3 s before the counter jump to the jump:
  * robot boxes (dets.json of the in-memory run) with team labels; dashed white = no team; other alliance grey
  * launch arcs (result shots) launched within [t-1.2, t+0.1] s, coloured by credited team, cross at the launch
  * every flight track alive at t (trail of the last 0.6 s) in grey
  * header: scouted vs credited split of the burst, launches per robot in the burst's kernel window, scout taps
Writes, per burst, <tag>_full.jpg (whole frame at the middle time, 1280 wide), <tag>_ann.jpg (crop rows over
time, annotated | raw side by side). Crop: --crop VIEW:x0,y0,x1,y1 (full-frame px) or automatic around the
scouted and the credited top robots.

  .venv/Scripts/python data/exp_visual_qa/tools/render_burst.py SEL.json [--only TAG ...] [--times ...]
SEL.json: [{"tag": ..., "match": ..., "alliance": ..., "t0": ..., "t1": ..., "crop": [..] optional, "times": [..]}]
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import attribution as A, scouting, views  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402
from find_bursts import bursts_for  # noqa: E402

CACHE = ROOT / "data/exp_visual_qa/cache"
IMG = ROOT / "data/exp_visual_qa/img"
TEAM_COL = [(255, 255, 0), (255, 0, 255), (0, 255, 0)]  # BGR: cyan, magenta, green for the alliance's 3 teams
UNL = (255, 255, 255)
OTHER = (140, 140, 140)
TRK = (175, 175, 175)


def dashed_rect(img, p0, p1, col, th=2, dash=8):
    x0, y0 = p0
    x1, y1 = p1
    for (a, b) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        n = max(1, int(np.hypot(b[0] - a[0], b[1] - a[1]) / dash))
        for i in range(0, n, 2):
            s = (int(a[0] + (b[0] - a[0]) * i / n), int(a[1] + (b[1] - a[1]) * i / n))
            e = (int(a[0] + (b[0] - a[0]) * min(i + 1, n) / n), int(a[1] + (b[1] - a[1]) * min(i + 1, n) / n))
            cv2.line(img, s, e, col, th)


def save_jpg(path, img, max_kb=290):
    while True:
        for q in (85, 78, 70, 62, 55):
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if len(buf) <= max_kb * 1024:
                Path(path).write_bytes(buf.tobytes())
                return len(buf) // 1024
        img = cv2.resize(img, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)  # shrink and retry


class MatchCtx:
    def __init__(self, key):
        self.key = key
        d = CACHE / key
        self.res = json.loads((d / "result.json").read_text())
        self.dets = json.loads((d / "dets.json").read_text())
        with open(d / "tracks.pkl", "rb") as fh:
            self.tracks = pickle.load(fh)
        self.meta = json.loads((d / "meta.json").read_text())
        self.vid = Video(video_path(key))
        self.vl = {v.name: v for v in views.PRODUCTIONS[self.meta["prod"]]}
        self.off = {n: v.offset(1920, 1080) for n, v in self.vl.items()}
        self.det_t = {}
        for row in self.dets:
            self.det_t.setdefault(row[1], {}).setdefault(row[0], []).append(row)
        self.det_keys = {vw: np.array(sorted(ts)) for vw, ts in self.det_t.items()}
        # track index: view -> list of (t_first, t_last, pts array)
        self.trk = {vw: [(p[0][0], p[-1][0], np.array(p)) for p in lst] for vw, lst in self.tracks.items()}

    def dets_at(self, t, tol=0.12):
        out = []
        for vw, keys in self.det_keys.items():
            if not len(keys):
                continue
            j = int(np.argmin(np.abs(keys - t)))
            if abs(keys[j] - t) <= tol:
                out += self.det_t[vw][keys[j]]
        return out


def draw(ctx, frame, t, al, teams, shots_al, shots_other):
    img = frame.copy()
    col = {tm: TEAM_COL[i] for i, tm in enumerate(teams)}
    # grey tracks alive at t: trail over the last 0.6 s
    for vw, lst in ctx.trk.items():
        ox, oy = ctx.off[vw]
        for ta, tb, p in lst:
            if tb < t - 0.6 or ta > t:
                continue
            q = p[(p[:, 0] <= t + 1e-3) & (p[:, 0] >= t - 0.6)]
            if len(q) >= 2:
                cv2.polylines(img, [np.int32(q[:, 1:3] + (ox, oy)).reshape(-1, 1, 2)], False, TRK, 1, cv2.LINE_AA)
            if len(q):
                cv2.circle(img, (int(q[-1, 1] + ox), int(q[-1, 2] + oy)), 5, TRK, 1, cv2.LINE_AA)
    # boxes
    for row in ctx.dets_at(t):
        _, vw, x0, y0, x1, y1, dal, team, seg, _fill = row
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        if dal != al:
            cv2.rectangle(img, p0, p1, OTHER, 1)
            cv2.putText(img, f"{team}", (p0[0], p0[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, OTHER, 1, cv2.LINE_AA)
            continue
        if team is None:
            dashed_rect(img, p0, p1, UNL, 2)
            cv2.putText(img, f"? s{seg}", (p0[0], p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, UNL, 2, cv2.LINE_AA)
        else:
            c = col.get(team, UNL)
            cv2.rectangle(img, p0, p1, c, 2)
            cv2.putText(img, f"{team} s{seg}", (p0[0], p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2, cv2.LINE_AA)
    # launch arcs
    for s, other in [(s, False) for s in shots_al] + [(s, True) for s in shots_other]:
        if not (t - 1.2 <= s["t"] <= t + 0.1):
            continue
        ox, oy = ctx.off[s["view"]]
        p = np.array([(q[1] + ox, q[2] + oy) for q in s["pts"]])
        c = (0, 128, 255) if other else col.get(s["team"], UNL)
        cv2.polylines(img, [np.int32(p).reshape(-1, 1, 2)], False, c, 1 if other else 2, cv2.LINE_AA)
        cv2.drawMarker(img, (int(p[0, 0]), int(p[0, 1])), c, cv2.MARKER_CROSS, 14, 2)
    return img


def header(lines, width, h=24):
    band = np.full((h * len(lines) + 8, width, 3), 20, np.uint8)
    for i, ln in enumerate(lines):
        cv2.putText(band, ln, (8, h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
    return band


def auto_crop(ctx, b, times):
    """Box around the scouted top and the credited top robot (main view preferred), plus launches."""
    teams = b["teams"]
    want = {b["scout_top"], b["credit_top"]}
    pts = {}
    for t in np.linspace(min(times), max(times), 8):
        for row in ctx.dets_at(t):
            if row[7] in want and row[6] == b["alliance"]:
                pts.setdefault(row[1], []).append(row[2:6])
    for s in ctx.res["shots"]:
        if s["alliance"] == b["alliance"] and b["t0"] - 5.5 <= s["t"] <= b["t1"] - 0.8 and s["team"] in want:
            ox, oy = ctx.off[s["view"]]
            pts.setdefault(s["view"], []).append([s["u"] + ox - 20, s["v"] + oy - 20, s["u"] + ox + 20, s["v"] + oy + 20])
    if not pts:
        return "main", list(ctx.vl["main"].box)
    vw = "main" if "main" in pts else max(pts, key=lambda k: len(pts[k]))
    B = np.array(pts[vw])
    x0, y0 = np.percentile(B[:, 0], 5), np.percentile(B[:, 1], 5)
    x1, y1 = np.percentile(B[:, 2], 95), np.percentile(B[:, 3], 95)
    bh = max(y1 - y0, 60)
    x0, x1 = x0 - 0.6 * bh, x1 + 0.6 * bh
    y0, y1 = y0 - 2.5 * bh, y1 + 0.3 * bh  # room for arcs above
    vx0, vy0, vx1, vy1 = ctx.vl[vw].box
    # aspect >= 4:3 wide
    w, h = x1 - x0, y1 - y0
    if w < 1.4 * h:
        cx = (x0 + x1) / 2
        x0, x1 = cx - 0.7 * h, cx + 0.7 * h
    x0, y0, x1, y1 = max(vx0, x0), max(vy0, y0), min(vx1, x1), min(vy1, y1)
    return vw, [int(x0), int(y0), int(x1), int(y1)]


def render(ctx, b, bm, tag, crop=None, times=None, n=4, panel_w=760):
    al, teams = b["alliance"], b["teams"]
    a0 = ctx.res["timeline"].get("auto_start") or 6.13
    if times is None:
        times = list(np.round(np.linspace(b["t0"] - 3.0, max(b["t0"], b["t1"] - 1.0), n), 2))
    shots_al = [s for s in ctx.res["shots"] if s["alliance"] == al]
    shots_other = [s for s in ctx.res["shots"] if s["alliance"] != al]
    # scout taps in video time (raw tap time, launch ~0.8 s earlier)
    taps = {}
    for tm in teams:
        rows = [x for x in bm[ctx.key].get(tm, []) if x.time_valid]
        taps[tm] = [sorted(round(float(x) + a0, 1) for x in r.shots if b["t0"] - 7 <= x + a0 <= b["t1"] + 1) for r in rows]
    win = [s for s in shots_al if b["t0"] - A.LAG_HI <= s["t"] <= b["t1"] - A.LAG_LO]
    by = {}
    for s in win:
        k = (s["team"], s["view"], s["how"])
        by[k] = by.get(k, 0) + 1
    lines = [f"{ctx.key} {al} burst {b['t0']:.1f}-{b['t1']:.1f}s (mt {b['mt0']}) K={b['K']}  "
             f"teams {teams} (cyan, magenta, green)",
             f"scouted share {b['sshare']}  credited share {b['cshare']}  moved {b['moved']}  "
             f"vis {list(b['vis'].values()) if b.get('vis') else '-'}  ev {list(b['ev'].values())}",
             "launches in window: " + ", ".join(f"{k[0]}/{k[1]}/{k[2]}:{v}" for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:8]),
             "scout taps (video s): " + " | ".join(f"{tm}: " + ";".join(
                 (",".join(f"{x:.1f}" for x in r[:12]) + ("..." if len(r) > 12 else "")) for r in taps[tm]) for tm in teams)]
    if crop is None:
        vw, crop = auto_crop(ctx, b, times)
    x0, y0, x1, y1 = crop
    rows = []
    tmid = times[len(times) // 2]
    for t in times:
        f = ctx.vid.frame_at(t)
        ann = draw(ctx, f, t, al, teams, shots_al, shots_other)
        if t == tmid:
            full = ann.copy()
        a_c, r_c = ann[y0:y1, x0:x1], f[y0:y1, x0:x1]
        s = panel_w / a_c.shape[1]
        a_c = cv2.resize(a_c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        r_c = cv2.resize(r_c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        lab = f"t={t:.2f} ({t - b['t0']:+.1f} s to jump)"
        cv2.putText(a_c, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        rows.append(np.hstack([a_c, np.full((a_c.shape[0], 4, 3), 20, np.uint8), r_c]))
    sheet = np.vstack([header(lines, rows[0].shape[1])] + rows)
    IMG.mkdir(parents=True, exist_ok=True)
    kb1 = save_jpg(IMG / f"{tag}_ann.jpg", sheet)
    cv2.rectangle(full, (x0, y0), (x1, y1), (0, 0, 255), 2)
    full = cv2.resize(full, (1280, 720), interpolation=cv2.INTER_AREA)
    full = np.vstack([header(lines[:2], 1280), full])
    kb2 = save_jpg(IMG / f"{tag}_full.jpg", full)
    print(tag, "crop", crop, "times", times, f"{kb1} KB / {kb2} KB")
    for ln in lines:
        print("  ", ln)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sel")
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    sel = json.loads(Path(a.sel).read_text())
    if a.only:
        sel = [s for s in sel if s["tag"] in a.only]
    bm = scouting.by_match(scouting.load())
    ctxs = {}
    for sp in sel:
        key = sp["match"]
        if key not in ctxs:
            ctxs.clear()
            ctxs[key] = MatchCtx(key)
            ctxs[key].bursts = bursts_for(key, ctxs[key].res, bm, ctxs[key].dets)
        ctx = ctxs[key]
        b = min((x for x in ctx.bursts if x["alliance"] == sp["alliance"]), key=lambda x: abs(x["t0"] - sp["t0"]))
        render(ctx, b, bm, sp["tag"], crop=sp.get("crop"), times=sp.get("times"), n=sp.get("n", 4),
               panel_w=sp.get("panel_w", 760))
