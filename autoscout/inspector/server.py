"""Match inspector: a local web app for browsing processed matches and diagnosing them.

    python -m autoscout.inspector            # then open http://localhost:8765

Serves the single-page UI (static/), a small JSON API built from output/<match>/ files, and
the downloaded match videos with HTTP range support (so the browser can seek).

API
  GET /api/matches                 summary row + diagnostic flags for every processed match
  GET /api/match/<key>             everything the UI draws for one match
  GET /api/match/<key>/ocr         bumper-OCR reads per detection (loads the cached scan)
  GET /video/<key>                 the match video (range requests)
  GET /frame/<key>?t=&w=           one video frame as a small JPEG (cached in data/cache/frames)
  GET /docs/<file>                 docs/ (architecture diagrams)
  GET /api/match/<key>/bursts      scoring bursts to label (alliance counter jumps)
  GET /api/labels/<key>?labeller=  one person's shooter labels for a match
  POST /api/labels/<key>           save {labeller, labels} (data/labels/<key>__<labeller>.json)
"""
from __future__ import annotations

import json
import mimetypes
import pickle
import re
import threading
import traceback
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from .. import config, labels, overlay, views
from ..video import video_path

STATIC = Path(__file__).resolve().parent / "static"
DOCS = config.ROOT / "docs"
FRAMES = config.CACHE / "frames"
KEY_RE = re.compile(r"^[0-9a-z_]+$")

_lock = threading.Lock()


# ------------------------------------------------------------------ data helpers
def _match_dirs() -> list[Path]:
    return sorted(p.parent for p in config.OUTPUT.glob("*/result.json"))


@lru_cache(maxsize=16)
def _opr(event: str) -> dict:
    try:
        from .. import opr
        return {str(k): v for k, v in opr.fuel_opr(event).items()}
    except Exception:
        return {}


@lru_cache(maxsize=256)
def _tba(key: str) -> dict:
    try:
        from .. import tba
        m = tba.match(key)
        return {"red": m.red, "blue": m.blue, "event": m.event_key, "comp_level": m.comp_level,
                "hub": {al: m.hub(al) for al in ("red", "blue")},
                "score": {"red": m.red_score, "blue": m.blue_score}, "youtube": m.youtube}
    except Exception:
        return {"red": [], "blue": [], "event": key.split("_")[0], "comp_level": "", "hub": {}, "score": {}}


def _identity_stats(res: dict, tba: dict) -> dict:
    """OCR agreement with the assigned identities, and segments whose own reads disagree."""
    agree = mass = 0.0
    contradicted = []
    for al in ("red", "blue"):
        teams = tba.get(al) or []
        for sid, s in res.get("identity", {}).get(al, {}).items():
            ev = np.array(s.get("ocr", [0, 0, 0])) + np.array(s.get("held", [0, 0, 0]))
            if s.get("team") is None or not teams or s["team"] not in teams:
                continue
            ti = teams.index(s["team"])
            agree += ev[ti]
            mass += ev.sum()
            if ev.sum() >= 2 and int(np.argmax(ev)) != ti:
                contradicted.append({"seg": int(sid), "alliance": al, "t0": s["t0"], "t1": s["t1"],
                                     "assigned": s["team"], "ocr_says": teams[int(np.argmax(ev))],
                                     "evidence": ev.round(2).tolist()})
    return {"agreement": (agree / mass) if mass else None, "evidence": mass, "contradicted": contradicted}


def _flags(res: dict, tba: dict, ident: dict) -> list[dict]:
    """Things worth a look, most severe first."""
    out = []
    for al in ("red", "blue"):
        meta = res["teams"].get(f"_{al}", {})
        tot, off = meta.get("timeline_total"), meta.get("tba_total")
        if meta.get("source") != "overlay":
            out.append({"level": "warn", "text": f"{al}: no overlay counter, total from {meta.get('source')}"})
        elif off and tot is not None and abs(tot - off) / max(off, 1) > 0.08:
            out.append({"level": "bad", "text": f"{al}: overlay read {tot:.0f} vs TBA {off} "
                                                f"({100 * (tot - off) / max(off, 1):+.0f} %)"})
    if ident["agreement"] is not None and ident["agreement"] < 0.85:
        out.append({"level": "bad" if ident["agreement"] < 0.75 else "warn",
                    "text": f"identity: only {100 * ident['agreement']:.0f} % of bumper reads agree with labels"})
    if ident["contradicted"]:
        out.append({"level": "warn", "text": f"{len(ident['contradicted'])} track segments labelled against "
                                             f"their own bumper reads"})
    seen = {}
    for al in ("red", "blue"):
        for s in res.get("identity", {}).get(al, {}).values():
            if s.get("team") is not None:
                seen[s["team"]] = seen.get(s["team"], 0) + (s["t1"] - s["t0"])
    for al in ("red", "blue"):
        for t in tba.get(al) or []:
            v = res["teams"].get(str(t), {})
            if seen.get(t, 0) < 30:
                out.append({"level": "warn", "text": f"{t} tracked for only {seen.get(t, 0):.0f} s"})
            if v.get("fuel", 0) >= 25 and v.get("shots_detected", 0) == 0:
                out.append({"level": "warn", "text": f"{t} credited {v['fuel']:.0f} fuel with no launch seen"})
    for k, c in (res.get("calibration") or {}).items():
        if c.get("rms", 0) > 2.5:
            out.append({"level": "warn", "text": f"camera {k}: calibration error {c['rms']:.1f} px"})
    order = {"bad": 0, "warn": 1}
    return sorted(out, key=lambda f: order.get(f["level"], 2))


def _production(res: dict) -> str:
    return "chezy" if (res.get("timeline") or {}).get("layout") == "chezy" else "first"


def match_summary(d: Path) -> dict:
    res = json.loads((d / "result.json").read_text())
    key = res["match"]
    tba = _tba(key)
    ident = _identity_stats(res, tba)
    opr = _opr(tba["event"])
    teams = {}
    for t, v in res["teams"].items():
        if not str(t).startswith("_"):
            teams[t] = {"alliance": v["alliance"], "fuel": v["fuel"], "opr": opr.get(str(t))}
    alliances = {al: {k: res["teams"].get(f"_{al}", {}).get(k) for k in ("source", "timeline_total", "tba_total")}
                 for al in ("red", "blue")}
    return {"key": key, "event": tba["event"], "comp_level": tba.get("comp_level"),
            "red": tba.get("red"), "blue": tba.get("blue"), "teams": teams, "alliances": alliances,
            "identity": ident["agreement"], "flags": _flags(res, tba, ident),
            "has_video": video_path(key).exists(),
            "n_bursts": len(labels.bursts(res)), "labels": labels.progress(key)}


def match_bursts(key: str) -> dict:
    res = json.loads((config.OUTPUT / key / "result.json").read_text())
    tba = _tba(key)
    return {"key": key, "bursts": labels.bursts(res), "teams": {al: tba.get(al) or [] for al in ("red", "blue")}}


def match_detail(key: str) -> dict:
    d = config.OUTPUT / key
    res = json.loads((d / "result.json").read_text())
    dets = json.loads((d / "dets.json").read_text()) if (d / "dets.json").exists() else []
    tba = _tba(key)
    ident = _identity_stats(res, tba)
    prod = _production(res)
    vlist = views.PRODUCTIONS[prod]
    lay = overlay.LAYOUTS.get((res.get("timeline") or {}).get("layout") or "")
    opr = _opr(tba["event"])
    try:
        from .. import tba as tbamod
        inactive = {k: sorted(v) for k, v in tbamod.inactive_periods(tbamod.match(key)).items()}
    except Exception:
        inactive = {"red": [], "blue": []}
    return {
        "key": key, "tba": tba, "opr": opr, "production": prod, "inactive": inactive,
        "views": [{"name": v.name, "box": v.box} for v in vlist],
        "overlay_rois": ({k: v for k, v in lay.rois.items() if k.endswith("_cnt")} if lay else {}),
        "result": {k: res.get(k) for k in ("teams", "shots", "inferred_shots", "identity", "timeline", "paths",
                                            "fuel_log",
                                            "calibration", "n_dets", "n_segments", "runtime_s")},
        "dets": dets,
        "identity_stats": ident,
        "flags": _flags(res, tba, ident),
        "field": {"length": config.FIELD_LENGTH, "width": config.FIELD_WIDTH,
                  "blue_hub": config.BLUE_HUB, "red_hub": config.RED_HUB, "zone_depth": 158.6 * 0.0254},
    }


def match_ocr(key: str) -> dict:
    """{det index: [[text, conf], ...]} for detections whose bumper crop was read."""
    pk = config.OUTPUT / key / "scan.pkl"
    if not pk.exists():
        return {}
    with _lock, open(pk, "rb") as fh:
        md = pickle.load(fh)
    cache = getattr(md, "ocr_cache", None) or {}
    out = {}
    for i, det in enumerate(md.dets):
        cid = getattr(det, "crop_id", None)
        if cid is not None and cid in cache:
            out[str(i)] = [[str(t), round(float(c), 3)] for t, c in cache[cid]]
    return out


def match_tracks(key: str) -> dict:
    """Every flight track the fuel scan kept, not only the launches: {view: [[t, u, v, t, u, v, ...], ...]} in
    view pixels (like shots' pts), for the inspector's "All flight tracks" layer."""
    pk = config.OUTPUT / key / "scan.pkl"
    if not pk.exists():
        return {}
    with _lock, open(pk, "rb") as fh:
        md = pickle.load(fh)
    out = {}
    for view, tracks in (getattr(md, "tracks_all", None) or {}).items():
        out[view] = [[x for p in tr.pts for x in (round(float(p[0]), 2), int(round(p[1])), int(round(p[2])))]
                     for tr in tracks]
    return out


def match_entries(key: str) -> dict:
    """Counted balls re-timed to when they dropped into the hub (autoscout/hubentry.py: the hub counts at most
    ~15 balls/s, so the scoreboard runs seconds behind in a dump): {alliance: [[count_t, entry_t, balls, conf]]}."""
    from .. import hubentry as H

    d = config.OUTPUT / key
    if not (d / "scan.pkl").exists():
        return {}
    tl = json.loads((d / "result.json").read_text())["timeline"]
    a0 = tl.get("auto_start") or 6.13
    t0, t1 = a0 - 1.0, (tl.get("teleop_start") or a0 + 25.1) + config.TELEOP_LEN + 6.0
    with _lock, open(d / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    tables = {v: H.track_ends(trs, md.cams.get(v)) for v, trs in md.tracks_all.items()}
    hf = {k: np.asarray(v, dtype=np.float64).reshape(-1, 3) for k, v in (md.hub_feats or {}).items()}
    del md
    out = {}
    for al in ("red", "blue"):
        inc, prev = [], 0
        for t, n in tl.get("counts", {}).get(al) or []:
            if n > prev and t0 <= t <= t1:
                inc.append((t, float(n - prev)))
            prev = max(prev, n)
        if not inc:
            continue
        rt, rate = H.entry_series(tables, hf, al, t0, t1)
        out[al] = [[round(e.count_t, 2), round(e.entry_t, 2), round(e.n, 2), round(e.conf, 2)]
                   for e in H.retime_queue(inc, rt, rate)]
    return out


def match_frame(key: str, t: float, w: int) -> Path | None:
    """The video frame at video time t, scaled to width w, as a cached JPEG (filmstrip, event board)."""
    t, w = round(max(0.0, t), 1), min(max(int(w), 120), 960)
    out = FRAMES / f"{key}_{t:.1f}_{w}.jpg"
    if out.exists():
        return out
    vp = video_path(key)
    if not vp.exists():
        return None
    import cv2
    cap = cv2.VideoCapture(str(vp))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
    finally:
        cap.release()
    if not ok:
        return None
    FRAMES.mkdir(parents=True, exist_ok=True)
    small = cv2.resize(fr, (w, round(fr.shape[0] * w / fr.shape[1])), interpolation=cv2.INTER_AREA)
    tmp = out.with_suffix(f".{threading.get_ident()}.jpg")
    cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
    tmp.replace(out)
    return out


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "autoscout-inspector"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, default=_default, separators=(",", ":")).encode()
        gz = len(body) > 1 << 16 and "gzip" in (self.headers.get("Accept-Encoding") or "")
        if gz:  # big payloads (every flight track of a match is several MB)
            import gzip
            body = gzip.compress(body, 5)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str | None = None):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        size = path.stat().st_size
        ctype = ctype or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:  # suffix range
                    start = max(0, size - int(m.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                return self._file(STATIC / "index.html", "text/html; charset=utf-8")
            if path.startswith("/static/"):
                f = (STATIC / path[len("/static/"):]).resolve()
                if STATIC.resolve() not in f.parents:
                    return self.send_error(403)
                return self._file(f)
            if path.startswith("/docs/"):
                f = (DOCS / path[len("/docs/"):]).resolve()
                if DOCS.resolve() not in f.parents:
                    return self.send_error(403)
                return self._file(f, "text/plain; charset=utf-8" if f.suffix == ".md" else None)
            if path == "/api/matches":
                rows = []
                for d in _match_dirs():
                    try:
                        rows.append(match_summary(d))
                    except Exception as e:
                        rows.append({"key": d.name, "error": repr(e)})
                return self._json(rows)
            m = re.match(r"^/api/match/([^/]+)(/ocr|/bursts|/tracks|/entries)?$", path)
            if m:
                key = m.group(1)
                if not KEY_RE.match(key) or not (config.OUTPUT / key / "result.json").exists():
                    return self._json({"error": "unknown match"}, 404)
                if m.group(2) == "/bursts":
                    return self._json(match_bursts(key))
                if m.group(2) == "/tracks":
                    return self._json(match_tracks(key))
                if m.group(2) == "/entries":
                    return self._json(match_entries(key))
                return self._json(match_ocr(key) if m.group(2) else match_detail(key))
            m = re.match(r"^/api/labels/([^/]+)$", path)
            if m and KEY_RE.match(m.group(1)):
                who = (parse_qs(urlparse(self.path).query).get("labeller") or [""])[0]
                if not labels.NAME_RE.match(who):
                    return self._json({"error": "labeller name required"}, 400)
                return self._json({**labels.load(m.group(1), who), "progress": labels.progress(m.group(1))})
            m = re.match(r"^/video/([^/]+)$", path)
            if m and KEY_RE.match(m.group(1)):
                return self._file(video_path(m.group(1)), "video/mp4")
            m = re.match(r"^/frame/([^/]+)$", path)
            if m and KEY_RE.match(m.group(1)):
                q = parse_qs(urlparse(self.path).query)
                try:
                    f = match_frame(m.group(1), float(q.get("t", ["0"])[0]), int(q.get("w", ["320"])[0]))
                except ValueError:
                    return self._json({"error": "bad t or w"}, 400)
                return self._file(f, "image/jpeg") if f else self.send_error(404)
            self.send_error(404)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": repr(e)}, 500)


    def do_POST(self):
        path = unquote(urlparse(self.path).path)
        try:
            m = re.match(r"^/api/labels/([^/]+)$", path)
            if not m or not KEY_RE.match(m.group(1)) or not (config.OUTPUT / m.group(1) / "result.json").exists():
                return self._json({"error": "unknown match"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 2_000_000:
                return self._json({"error": "bad body"}, 400)
            body = json.loads(self.rfile.read(n).decode("utf-8"))
            with _lock:
                doc = labels.save(m.group(1), body.get("labeller", ""), body.get("labels", {}))
            return self._json({"ok": True, "saved": len(doc["labels"]), "progress": labels.progress(m.group(1))})
        except ValueError as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            traceback.print_exc()
            self._json({"error": repr(e)}, 500)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def serve(port: int = 8765, open_browser: bool = False):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://localhost:{port}"
    print(f"auto-scout match inspector: {url}   (Ctrl+C to stop)", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
