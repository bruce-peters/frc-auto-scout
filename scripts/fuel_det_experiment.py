"""A/B: the flying-fuel YOLO as an extra blob source for launch tracking, on held-out matches.

  scan  redo the fuel step of each held-out match from its video, starting from the baseline
        snapshot of the cached scan (robot detections, identity crops and OCR are reused, so only
        the fuel step differs). Logs every centre frame's motion blobs (and, with --model, the
        flying-fuel detections at a low confidence) to data/exp_fueldet/blobs/<key>.pkl.
        --no-det runs the CPU-only motion part; `dets` then adds the detections (GPU).
  eval  re-track offline per variant (motion only = baseline; motion + detections at several
        confidences), run pipeline.analyse, and compare the RESULTS.md team-level numbers vs
        fuel OPR (paired bootstrap over matches) -> data/exp_fueldet/ab.json.

The baseline snapshot is data/exp_fueldet/baseline/<key>/scan.pkl (copied from output/).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate  # noqa: E402
from autoscout import config, fuel, opr, pipeline  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402

EXP = config.DATA / "exp_fueldet"


def heldout_keys() -> list[str]:
    return (EXP / "heldout.txt").read_text().split()


def load_base(key: str):
    """The baseline snapshot of a held-out match; other matches are read from output/ (read only)."""
    p = EXP / "baseline" / key / "scan.pkl"
    with open(p if p.exists() else config.OUTPUT / key / "scan.pkl", "rb") as fh:
        return pickle.load(fh)


def cmd_scan(keys: list[str], model: str | None, conf: float, imgsz: int):
    opts = pipeline.Options(flying_fuel_model=model, flying_fuel_conf=conf, flying_fuel_imgsz=imgsz)
    (EXP / "blobs").mkdir(exist_ok=True)
    for key in keys:
        out = EXP / "blobs" / f"{key}.pkl"
        if out.exists():
            print("skip", key, flush=True)
            continue
        tic = time.time()
        md = load_base(key)
        video = Video(video_path(key))
        fs = pipeline.rescan_fuel(md, video, opts, keep_blobs=True)
        with open(out, "wb") as fh:
            pickle.dump({"fps": video.fps, "fuel_step": opts.fuel_step, "model": model, "conf": conf,
                         "blob_log": fs.blob_log, "hub_feats": md.hub_feats}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        nd = sum(len(r[2]) for rows in fs.blob_log.values() for r in rows)
        nm = sum(len(r[1]) for rows in fs.blob_log.values() for r in rows)
        print(f"{key}: {time.time() - tic:.0f}s motion blobs {nm}, detections {nd}", flush=True)


def cmd_dets(keys: list[str], model: str, conf: float, imgsz: int):
    """Add flying-fuel detections to motion-only blob logs (same centre frames)."""
    from autoscout import views

    det = pipeline.FlyingFuelDetector(model, imgsz, conf)
    for key in keys:
        bp = EXP / "blobs" / f"{key}.pkl"
        with open(bp, "rb") as fh:
            log = pickle.load(fh)
        if log.get("model"):
            print("skip (has detections)", key, flush=True)
            continue
        tic = time.time()
        md = load_base(key)
        video = Video(video_path(key))
        W, H = video.width, video.height
        vlist = views.PRODUCTIONS[md.prod]
        vbox = {v.name: v.scaled(W, H) for v in vlist}
        valid = {}
        for v in vlist:
            x0, y0, x1, y1 = vbox[v.name]
            m = np.ones((y1 - y0, x1 - x0), np.uint8)
            for mx0, my0, mx1, my1 in v.mask_boxes(W, H):
                m[max(0, my0):max(0, my1), max(0, mx0):max(0, mx1)] = 0
            valid[v.name] = m
        # centre frame index -> row of each view's log
        rows_at: dict[int, list] = {}
        for vn, rows in log["blob_log"].items():
            for i, r in enumerate(rows):
                rows_at.setdefault(int(round(r[0] * video.fps)), []).append((vn, i))
        start, end = int(pipeline.md_t0(md) * video.fps), int(pipeline.md_t1(md) * video.fps)
        nd = 0
        for idx, frame in video.iter_frames(start, end, step=log["fuel_step"]):
            if idx not in rows_at:
                continue
            dets = det(frame)
            for vn, i in rows_at[idx]:
                tc, mb, _ = log["blob_log"][vn][i]
                vd = fuel.view_detections(dets, vbox[vn], valid[vn])
                log["blob_log"][vn][i] = (tc, mb, vd)
                nd += len(vd)
        log.update(model=model, conf=conf)
        with open(bp, "wb") as fh:
            pickle.dump(log, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{key}: {time.time() - tic:.0f}s detections {nd}", flush=True)


def retrack(blob_log: dict, fuel_step: int, conf: float | None, det_only: bool = False) -> dict:
    tracks = {}
    for vn, rows in blob_log.items():
        tr = fuel.FlightTracker(gate=45.0 * fuel_step ** 0.7)
        for tc, mb, dd in rows:
            blobs = [] if det_only else [fuel.Blob(*map(float, r)) for r in mb]
            if conf is not None:
                blobs = fuel.merge_detections(blobs, dd, conf)
            tr.update(tc, blobs)
        tracks[vn] = tr.flush()
    return tracks


def confirmed(tracks: dict, blob_log: dict, conf: float) -> dict:
    """Keep only tracks with at least one point inside a detection box (conf >= `conf`)."""
    out = {}
    for vn, trs in tracks.items():
        dets = {tc: dd[dd[:, 4] >= conf] for tc, _, dd in blob_log[vn] if len(dd)}
        keep = []
        for tr in trs:
            for t, u, v, _ in tr.pts:
                d = dets.get(t)
                if d is not None and len(d) and np.any((u >= d[:, 0] - 2) & (u <= d[:, 2] + 2) &
                                                       (v >= d[:, 1] - 2) & (v <= d[:, 3] + 2)):
                    keep.append(tr)
                    break
        out[vn] = keep
    return out


def same_tracks(a: dict, b: dict) -> bool:
    for vn in a:
        ta, tb = a[vn], b.get(vn, [])
        if len(ta) != len(tb):
            return False
        for x, y in zip(ta, tb):
            if len(x.pts) != len(y.pts) or not np.allclose(np.array(x.pts), np.array(y.pts)):
                return False
    return True


def shot_stats(res: dict) -> dict:
    """Detected launches per alliance vs balls the alliance scored (overlay counter)."""
    out = {}
    for al in ("red", "blue"):
        teams = [v for k, v in res["teams"].items() if not k.startswith("_") and v["alliance"] == al]
        out[al] = (sum(t["shots_detected"] for t in teams), res["teams"][f"_{al}"]["timeline_total"] or 0)
    return out


def bootstrap(rows_a: list[dict], rows_b: list[dict], n: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over matches of (b - a) for corr and top-scorer agreement."""
    rng = np.random.default_rng(seed)
    keys = sorted({r["match"] for r in rows_a})
    ga = {k: [r for r in rows_a if r["match"] == k] for k in keys}
    gb = {k: [r for r in rows_b if r["match"] == k] for k in keys}
    dc, dt = [], []
    for _ in range(n):
        pick = rng.choice(keys, len(keys))
        ra, rb = [], []
        for i, k in enumerate(pick):  # keep resampled copies distinct alliances
            ra += [{**r, "match": f"{k}#{i}"} for r in ga[k]]
            rb += [{**r, "match": f"{k}#{i}"} for r in gb[k]]
        ma, mb = evaluate.team_metrics(ra), evaluate.team_metrics(rb)
        dc.append(mb["corr"] - ma["corr"])
        dt.append(mb["top_pct"] - ma["top_pct"])
    dc, dt = np.array(dc), np.array(dt)
    return {"d_corr_ci90": np.percentile(dc, [5, 95]).round(3).tolist(), "p_corr_better": float((dc > 0).mean()),
            "d_top_ci90": np.percentile(dt, [5, 95]).round(1).tolist(), "p_top_better": float((dt > 0).mean())}


def cmd_eval(keys: list[str], confs: list[float], det_only: list[float], publish: bool = False,
             confirm_confs: list[float] = ()):
    from autoscout import tba

    opts = pipeline.Options()
    # zone_only: no launch evidence at all (split by alliance-zone occupancy) - a reference arm
    variants = ([("cached", None), ("motion", None), ("zone_only", None)] + [(f"union@{c:g}", c) for c in confs] +
                [(f"confirm@{c:g}", c) for c in confirm_confs])
    variants += [(f"det_only@{c:g}", c) for c in det_only]
    rows = {name: [] for name, _ in variants}
    shots = {name: [] for name, _ in variants}
    oprs = {}
    reproduced = 0
    for key in keys:
        bp = EXP / "blobs" / f"{key}.pkl"
        if not bp.exists():
            print("no blob log:", key)
            continue
        with open(bp, "rb") as fh:
            log = pickle.load(fh)
        md = load_base(key)
        cached = md.tracks_all
        info = tba.match(key)
        if info.event_key not in oprs:
            oprs[info.event_key] = opr.fuel_opr(info.event_key)
        for name, conf in variants:
            if name == "cached":
                md.tracks_all = cached
            elif name == "motion":
                md.tracks_all = motion = retrack(log["blob_log"], log["fuel_step"], None)
                reproduced += same_tracks(md.tracks_all, cached)
            elif name == "zone_only":
                md.tracks_all = {vn: [] for vn in cached}
            elif name.startswith("confirm"):
                md.tracks_all = confirmed(motion, log["blob_log"], conf)
            else:
                md.tracks_all = retrack(log["blob_log"], log["fuel_step"], conf, det_only=name.startswith("det_only"))
            res = json.loads(json.dumps(pipeline.analyse(md, opts), default=pipeline._json_default))
            rows[name] += evaluate.team_rows_of(res, info, oprs[info.event_key])
            shots[name].append(shot_stats(res))
        print(key, flush=True)
    print(f"\nmotion-only re-track reproduces the cached tracks in {reproduced}/{len(shots['cached'])} matches")
    out = {"matches": len(shots["cached"]), "reproduced": reproduced, "variants": {}}
    events = sorted({r["match"].split("_")[0] for r in rows["cached"]})
    print(f"\n{'variant':16s} {'corr':>6s} {'top':>9s} {'mae':>6s} {'shots/ball':>10s}  " +
          "  ".join(f"{e}: corr top" for e in events))
    for name, _ in variants:
        m = evaluate.team_metrics(rows[name])
        s = np.array([[v[0], v[1]] for st in shots[name] for v in st.values()], float)
        m["shots_per_ball"] = float(s[:, 0].sum() / max(1, s[:, 1].sum()))
        m["by_event"] = {e: evaluate.team_metrics([r for r in rows[name] if r["match"].startswith(e + "_")])
                         for e in events}
        if name != "cached":
            m["vs_cached"] = bootstrap(rows["cached"], rows[name])
        out["variants"][name] = m
        ev_s = "  ".join(f"{e}: {m['by_event'][e]['corr']:.2f} {m['by_event'][e]['top_pct']:3.0f}%" for e in events)
        print(f"{name:16s} {m['corr']:6.3f} {m['top']:3d}/{m['alliances']:3d} {m['mae']:6.1f} "
              f"{m['shots_per_ball']:10.2f}  {ev_s}")
    for name in out["variants"]:
        if "vs_cached" in out["variants"][name]:
            print(name, out["variants"][name]["vs_cached"])
    out["events"] = events
    (EXP / "ab.json").write_text(json.dumps(out, indent=1))
    if publish:  # read by scripts/evaluate.py for the RESULTS.md section
        (EXP / "ab_published.json").write_text(json.dumps(out, indent=1))
    with open(EXP / "ab_rows.pkl", "wb") as fh:
        pickle.dump(rows, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["scan", "dets", "eval"])
    ap.add_argument("--no-det", action="store_true", help="scan: motion blobs only (CPU)")
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--model", default=str(config.MODELS / "fuel_flying.pt"))
    ap.add_argument("--conf", type=float, default=0.05, help="scan: lowest detector confidence logged")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--confs", type=float, nargs="*", default=[0.15, 0.25, 0.4, 0.6])
    ap.add_argument("--confirm-confs", type=float, nargs="*", default=[0.1, 0.25],
                    help="eval: motion-only tracks kept only when a detection confirms one point")
    ap.add_argument("--det-only", type=float, nargs="*", default=[0.15, 0.25],
                    help="eval: tracking on detections alone, at these confidences")
    ap.add_argument("--publish", action="store_true",
                    help="eval: also write data/exp_fueldet/ab_published.json (RESULTS.md section)")
    a = ap.parse_args()
    keys = a.matches or heldout_keys()
    si, sn = map(int, a.shard.split("/"))
    keys = keys[si::sn]
    if a.cmd == "scan":
        cmd_scan(keys, None if a.no_det else a.model, a.conf, a.imgsz)
    elif a.cmd == "dets":
        cmd_dets(keys, a.model, a.conf, a.imgsz)
    else:
        cmd_eval(keys, a.confs, a.det_only, a.publish, a.confirm_confs)


if __name__ == "__main__":
    main()
