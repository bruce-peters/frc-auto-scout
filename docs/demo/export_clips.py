"""Build the replay demo page: 30 s clips of chosen matches with their overlay data and robot photos.

  python docs/demo/export_clips.py [--elim] [--data-only] [--no-crops] [match keys...]

Writes dist/demo/: <name>.mp4 (cut with ffmpeg from data/videos), <name>.json (boxes, launch arcs, inferred
marks, scoreboard, per-robot credit, field paths), crops/*.jpg (robot photos whose bumper number was read) and
clips.json, and copies auto-scout-replays.html there; publish that folder as a static page. Needs the full data
pack and the match videos (scripts/setup.py --full). Never add scouting data to it.
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autoscout import config, field, pipeline, tba, views  # noqa: E402

OUT = ROOT / "dist" / "demo"  # the page, its clips, overlay data and robot photos (git-ignored)
OUT.mkdir(parents=True, exist_ok=True)
W, H = 1920, 1080
DUR = 30.0
RED_COLS = ["#f2471c", "#ff9a3c", "#ff7a9c"]
BLUE_COLS = ["#7ea6d4", "#5cc4ae", "#d6dce4"]
EVENTS = {"cc": "Chezy Champs", "joh": "Championship, Johnson Division", "cancmp": "NorCal State Championship"}
CLIPS = [  # (match, focus team or None for the busiest 30 s, note)
    ("2026cc_qm54", 5026, "5026 with 254 and 604"),
    ("2026cc_qm17", 6800, "6800 shoots from right under the blue end camera"),
    ("2026cc_qm49", 254, "254 and 6800 against 2910"),
    ("2026cc_qm21", None, "973 and 687 against 604"),
    ("2026joh_sf1m1", 5026, "5026 in the division playoffs"),
    ("2026joh_qm14", 5026, "5026 with 3620 and 117"),
    ("2026joh_sf7m1", 5026, "5026 against 118 and 190"),
    ("2026cancmp_qm5", 5026, "5026's biggest scoring match"),
    ("2026cancmp_sf4m1", 5026, "5026 with 8 and 841 in the playoffs"),
    ("2026cancmp_qm107", None, "254 and 846 against 5026"),
]


def match_name(key):
    ev, m = key[4:].split("_")
    import re
    q = re.fullmatch(r"qm(\d+)", m)
    sf = re.fullmatch(r"sf(\d+)m\d+", m)
    f = re.fullmatch(r"f1m(\d+)", m)
    label = f"Qual {q.group(1)}" if q else f"Playoff {sf.group(1)}" if sf else f"Final {f.group(1)}" if f else m
    return EVENTS.get(ev, ev), label


ELIM = "--elim" in sys.argv
if ELIM:
    pipeline.ELIM_ARCS = pipeline.ELIM_BOXES = True
    pipeline.IDENT_FAR_COST = 20.0


def r3(x):
    return round(float(x), 4)


def robot_crops(md, key, name, teams, clock, ident):
    """Up to 3 photos per robot from the whole match: its largest unobstructed boxes, a bumper read of its own
    number counting extra, at least 12 s apart and at most 2 from one camera; cut from the 1080p source. A photo
    needs its own bumper read of the number, or a track whose bumper reads clearly back that team (a mislabelled
    track would show the wrong robot); a robot with neither gets no photo."""
    import cv2
    rows, dets = md.det_rows, md.dets
    vr = {v.name: v.scaled(W, H) for v in views.PRODUCTIONS[md.prod]}
    by_frame = {}
    for i, r in enumerate(rows):
        by_frame.setdefault((r[1], r[0]), []).append(i)
    reads = getattr(md, "ocr_cache", {}) or {}
    (OUT / "crops").mkdir(exist_ok=True)
    out = {}
    for al in teams:
        for team in teams[al]:
            cands = []
            for i, r in enumerate(rows):
                if r[7] != team:
                    continue
                x0, y0, x1, y1 = r[2:6]
                vx0, vy0, vx1, vy1 = vr[r[1]]
                if y1 - y0 < 60 or x0 < vx0 + 4 or x1 > vx1 - 4 or y0 < vy0 + 4 or y1 > vy1 - 4:
                    continue
                area = (x1 - x0) * (y1 - y0)
                crowded = False
                for j in by_frame[(r[1], r[0])]:
                    if j == i:
                        continue
                    a0, b0, a1, b1 = rows[j][2:6]
                    ov = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
                    if ov > 0.08 * area:
                        crowded = True
                        break
                if crowded:
                    continue
                cid = dets[i].crop_id
                ok = any(txt == str(team) and c >= 0.6 for txt, c in reads.get(cid, []) or [])
                if not ok:
                    ev = (ident.get(al) or {}).get(str(r[8]), {}).get("ocr")
                    ti = teams[al].index(team)
                    if not ev or ev[ti] < 2.0 or ev[ti] < max(e for k, e in enumerate(ev) if k != ti) + 2.0:
                        continue
                    others = [str(o) for a in teams for o in teams[a] if o != team]
                    if any(len(txt) >= 2 and c >= 0.2 and str(team).find(txt) < 0 and any(txt in o for o in others)
                           for txt, c in reads.get(cid, []) or []):
                        continue  # this frame's bumper read points at another robot
                cands.append(((y1 - y0) * (1.6 if ok else 1.0), i, ok))
            if any(ok for _, _, ok in cands):  # its own bumper read beats a track's word for it
                cands = [c for c in cands if c[2]]
            cands.sort(reverse=True)
            picked = []
            for sc, i, ok in cands:
                if len(picked) == 3:
                    break
                if any(abs(rows[i][0] - rows[k][0]) < 12 for k, _ in picked):
                    continue
                if sum(rows[k][1] == rows[i][1] for k, _ in picked) >= 2:
                    continue
                picked.append((i, ok))
            res = []
            for n, (i, ok) in enumerate(sorted(picked, key=lambda q: rows[q[0]][0])):
                r = rows[i]
                x0, y0, x1, y1 = r[2:6]
                vx0, vy0, vx1, vy1 = vr[r[1]]
                bw, bh = x1 - x0, y1 - y0
                cx0, cx1 = max(vx0, x0 - 0.18 * bw), min(vx1, x1 + 0.18 * bw)
                cy0, cy1 = max(vy0, y0 - 0.30 * bh), min(vy1, y1 + 0.08 * bh)
                png = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{r[0]:.3f}", "-i",
                                      str(ROOT / "data" / "videos" / f"{key}.mp4"), "-frames:v", "1", "-f", "image2pipe",
                                      "-vcodec", "png", "-"], capture_output=True, check=True).stdout
                img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                sx, sy = img.shape[1] / W, img.shape[0] / H
                crop = img[int(cy0 * sy):int(cy1 * sy), int(cx0 * sx):int(cx1 * sx)]
                if crop.size == 0:
                    continue
                k = 200 / crop.shape[0]
                crop = cv2.resize(crop, (max(1, int(crop.shape[1] * k)), 200), interpolation=cv2.INTER_AREA)
                fn = f"crops/{name}_{team}_{n}.jpg"
                cv2.imwrite(str(OUT / fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 82])
                res.append({"src": fn, "view": r[1], "clock": clock(r[0]), "read": ok,
                            "w": crop.shape[1], "h": crop.shape[0]})
            out[str(team)] = res
    return out


def export(key, focus, note):
    event, label = match_name(key)
    title = f"{event}, {label}"
    md = pickle.load(open(config.OUTPUT / key / "scan.pkl", "rb"))
    pipeline.load_dino(md, config.OUTPUT / key)
    pipeline.load_fuel60(md)  # 60 Hz flight tracks (pipeline.FUEL_60HZ)
    res = pipeline.analyse(md, pipeline.Options())
    rows = md.det_rows
    info = tba.match(key)
    tl = res["timeline"]
    a0, ts = tl.get("auto_start") or 6.13, tl.get("teleop_start")

    def to_vt(mt):
        return (ts + mt - 20) if ts and mt >= 20 else a0 + mt

    al_f = "red" if focus in info.red else "blue"
    offs = {v.name: v.offset(W, H) for v in views.PRODUCTIONS[md.prod]}
    shots = [dict(s, counted=True) for s in res["shots"]] + [dict(s, counted=False) for s in res.get("inferred_shots", [])]
    # window: the 30 s with the most launches by the focus team (counted or inferred), after auto starts
    st = np.array(sorted(s["t"] for s in shots if s["counted"] and (focus is None or (s["team"] or s.get("elim_team")) == focus)))
    best, t0 = -1, a0
    for c in np.arange(a0, (st.max() if len(st) else a0) + 1, 0.5):
        n = int(((st >= c + 3) & (st < c + DUR - 4)).sum())
        if n > best:
            best, t0 = n, float(c)
    name = key.replace("2026", "", 1)
    if "--data-only" in sys.argv and (OUT / f"{name}.json").exists():  # keep the window of the published video
        t0 = json.loads((OUT / f"{name}.json").read_text())["t0"]
    t1 = t0 + DUR
    mp4 = OUT / f"{name}.mp4"
    if "--data-only" not in sys.argv:
      subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{t0:.3f}", "-i", str(ROOT / "data" / "videos" / f"{key}.mp4"),
                    "-t", f"{DUR}", "-vf", "scale=1280:-2,fps=30", "-c:v", "libx264", "-crf", "29", "-preset", "slow",
                    "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(mp4)], check=True)
    teams = {"red": list(info.red), "blue": list(info.blue)}
    colors = {str(t): (RED_COLS if al == "red" else BLUE_COLS)[i % 3] for al in teams for i, t in enumerate(teams[al])}
    frames = {}
    for r in rows:
        if t0 - 0.2 <= r[0] <= t1 + 0.2:
            frames.setdefault(round(r[0] - t0, 3), []).append(
                [r3(r[2] / W), r3(r[3] / H), r3(r[4] / W), r3(r[5] / H), r[6][0], r[7], r[10] if len(r) > 10 else None])
    sh_out = []
    for s in shots:
        if not (t0 - 1.0 <= s["t"] <= t1):
            continue
        ox, oy = offs.get(s["view"], (0, 0))
        team = s["team"] if s["team"] is not None else s.get("elim_team")
        inferred = bool(s.get("inferred")) or (s["team"] is None and s.get("elim_team") is not None)
        sh_out.append({"t": round(s["t"] - t0, 3), "team": team, "inf": inferred, "counted": s["counted"],
                       "pts": [[round(p[0] - t0, 3), r3((p[1] + ox) / W), r3((p[2] + oy) / H)] for p in s["pts"]]})
    counts = {al: [[round(vt - t0, 2), int(c)] for vt, c in tl["counts"].get(al, [])] for al in ("red", "blue")}
    credit = [[round(to_vt(mt) + 2.0 - t0, 2), tm, round(b, 3)] for mt, tm, b in res["fuel_log"]]
    hub = {al: (info.hub(al) or {}).get("totalCount") for al in ("red", "blue")}
    # per robot, whole match: fuel (scaled to the official total), auto / teleop, launches seen (and inferred)
    robots = {}
    for al in teams:
        tot = sum(res["teams"][t]["fuel"] for t in teams[al]) or 1.0
        for t in teams[al]:
            v = res["teams"][t]
            mine = [s for s in res["shots"] if s["team"] == t]
            robots[str(t)] = {"fuel": round(v["fuel"]), "auto": round(v["fuel_by_period"].get("auto", 0.0)),
                              "teleop": round(v["fuel"] - v["fuel_by_period"].get("auto", 0.0)),
                              "launches": len(mine), "inferred": sum(bool(s.get("inferred")) for s in mine),
                              "share": round(v["fuel"] / tot, 3)}
    # field positions over the window (every 0.2 s) and where each launch left from
    paths, pos = {}, {}
    for t in info.red + info.blue:
        pts = [(to_vt(m) - t0, x, y) for m, x, y in (res["paths"].get(t) or res["paths"].get(str(t)) or []) if m is not None]
        pos[t] = np.array(pts) if pts else np.zeros((0, 3))
        paths[str(t)] = [[round(a, 2), round(x, 2), round(y, 2)] for a, x, y in pts[::2] if -8.0 <= a <= DUR + 1]
    for so in sh_out:
        p = pos.get(so["team"])
        if so["team"] is not None and p is not None and len(p):
            j = int(np.argmin(np.abs(p[:, 0] - so["t"])))
            if abs(p[j, 0] - so["t"]) < 0.8:
                so["fx"], so["fy"] = round(float(p[j, 1]), 2), round(float(p[j, 2]), 2)
    def clock(vt):
        mt = (20 + vt - ts) if ts and vt >= ts else vt - a0
        if mt < 20:
            return f"auto, {max(0, int(round(20 - mt)))} s left"
        left = max(0, int(round(160 - mt)))
        return f"teleop, {left // 60}:{left % 60:02d} left"
    if "--no-crops" not in sys.argv:
        for t_, c_ in robot_crops(md, key, name, teams, clock, res.get("identity") or {}).items():
            robots[t_]["crops"] = c_
    fld = {"L": round(field.L, 3), "W": round(field.W, 3), "zone": round(field.ALLIANCE_ZONE_DEPTH, 3),
           "hub": round(field.HUB_HALF, 3), "blueHub": [round(float(c), 3) for c in field.hub_center("blue")],
           "redHub": [round(float(c), 3) for c in field.hub_center("red")]}
    final = {str(t): v["fuel"] for t, v in res["teams"].items() if str(t).isdigit()}
    d = {"key": key, "event": event, "label": label, "title": title, "note": note, "focus": focus, "t0": round(t0, 3), "dur": DUR,
         "a0": round(a0 - t0, 3), "ts": None if ts is None else round(ts - t0, 3), "teams": teams, "colors": colors,
         "frames": sorted([[k, v] for k, v in frames.items()]), "shots": sh_out, "counts": counts, "credit": credit,
         "tba": hub, "final": final, "elim_counted": ELIM, "robots": robots, "paths": paths, "field": fld}
    (OUT / f"{name}.json").write_text(json.dumps(d, separators=(",", ":")))
    (OUT / f"{name}.meta.json").write_text(json.dumps({"name": name, "event": event, "label": label, "note": note,
                                                        "teams": teams, "colors": colors}))
    print(key, f"window {t0:.1f}-{t1:.1f} s, focus launches {best}, {mp4.stat().st_size / 1e6:.1f} MB video,"
          f" {(OUT / f'{name}.json').stat().st_size / 1e3:.0f} kB data,"
          f" inferred arcs {sum(s['inf'] for s in sh_out)}", flush=True)


only = [a for a in sys.argv[1:] if not a.startswith("--")]
for c in CLIPS:
    if not only or c[0] in only:
        export(*c)
(OUT / "clips.json").write_text(json.dumps(
    [json.loads((OUT / f"{c[0].replace('2026', '', 1)}.meta.json").read_text()) for c in CLIPS], separators=(",", ":")))

import shutil  # noqa: E402

shutil.copyfile(Path(__file__).resolve().parent / "auto-scout-replays.html", OUT / "auto-scout-replays.html")
