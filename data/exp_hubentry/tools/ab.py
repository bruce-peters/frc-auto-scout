"""A/B step 2: re-run the per-team split on the captured inputs (capture.py) with each credit variant, write
result.json per arm (base result with teams / fuel_log replaced), score every arm against 5026's scouting with
scripts/score_scouting.score, and paired-bootstrap each arm against today's split ("base") over matches.

Arms that keep attribution.split_increments monkeypatch its kernel / LAG constants; the entry arms call
autoscout.hubentry.split_entries with balls re-timed from per-match hub-entry evidence.

  .venv/Scripts/python data/exp_hubentry/tools/ab.py [--arms a,b,...] [--boot 2000]
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import zlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from autoscout import attribution as A, config, hubentry as H, scouting, tba  # noqa: E402
from score_scouting import score, to_mt_fn, tri  # noqa: E402

DST = config.DATA / "exp_hubentry"
AB = DST / "ab"


class TL:
    def __init__(self, a0, ts):
        self.auto_start, self.teleop_start = a0, ts

    def match_time(self, vt):
        if self.auto_start is None:
            return None
        if self.teleop_start is not None and vt >= self.teleop_start:
            return config.AUTO_LEN + (vt - self.teleop_start)
        return vt - self.auto_start


# ---------------------------------------------------------------- arms
def tri_kernel(lo, peak, hi, area=None):
    def k(dt):
        w = np.where(dt < peak, (dt - lo) / (peak - lo), (hi - dt) / (hi - peak))
        w = np.clip(w, 0, 1)
        return w * (area / (0.5 * (hi - lo))) if area else w
    return k


def patched_split(lo, peak, hi, area=None):
    """split_increments with attribution's kernel / window replaced."""
    def run(inc, teams, to_mt, log=None, opr=None, ctx=None):
        saved = (A.LAG_LO, A.LAG_PEAK, A.LAG_HI, A.kernel)
        A.LAG_LO, A.LAG_PEAK, A.LAG_HI = lo, peak, hi
        A.kernel = tri_kernel(lo, peak, hi, area)
        try:
            return A.split_increments(inc, teams, to_mt, log=log, opr=opr)
        finally:
            A.LAG_LO, A.LAG_PEAK, A.LAG_HI, A.kernel = saved
    return run


def base_split(inc, teams, to_mt, log=None, opr=None, ctx=None):
    return A.split_increments(inc, teams, to_mt, log=log, opr=opr)


def entry_split(w_track=1.0, w_blob=1.0, queue=True, kern=(H.FLIGHT_LO, H.FLIGHT, H.FLIGHT_HI), area=H.KERNEL_AREA,
                fallback_conf=0.0, const_lag=None, min_ev=0.3, d_first=(0.8, 2.6), d_min=0.6, mix=0.0):
    k = tri_kernel(kern[0], kern[1], kern[2], area)

    def run(inc, teams, to_mt, log=None, opr=None, ctx=None):
        if const_lag is not None:
            ents = [H.Retimed(t, t - H.READ_GAP / 2 - const_lag, n, 1.0) for t, n in inc if n > 0]
        else:
            rt, rate = H.entry_series(ctx["tables"], ctx["hub_feats"], ctx["al"], ctx["t0"], ctx["t1"],
                                      w_track=w_track, w_blob=w_blob)
            if queue:
                ents = H.retime_queue(inc, rt, rate, d_min=d_min, d_first=d_first, min_ev=min_ev)
            else:
                ents = H.retime(inc, rt, rate)
        model = A.load_model() if A.ATTR_MODEL else None
        return H.split_entries(inc, teams, to_mt, ents, log=log, opr=opr, model=model, kern=k,
                               flight=(kern[0], kern[2]), fallback_conf=fallback_conf, mix=mix)
    return run


ARMS = {
    "base": base_split,
    "k_peak2.45": patched_split(0.8, 2.45, 5.5),
    "k_narrow": patched_split(1.2, 2.45, 4.2, area=2.35),
    "k_narrow_raw": patched_split(1.2, 2.45, 4.2),
    "const_lag1.6": entry_split(const_lag=1.6),
    "entry_q": entry_split(),
    "entry_q_tracks": entry_split(w_blob=0.0),
    "entry_q_blobs": entry_split(w_track=0.0),
    "entry_simple": entry_split(queue=False),
    "entry_q_tight": entry_split(kern=(0.3, 0.8, 1.5)),
    "entry_q_wide": entry_split(kern=(0.0, 0.9, 2.5)),
    "entry_q_raw": entry_split(area=None),
    "entry_q_f1.05": entry_split(kern=(0.3, 1.05, 2.0)),
    "entry_q_f1.05_wide": entry_split(kern=(0.0, 1.05, 2.6)),
    "const_lag1.6_f1.05": entry_split(const_lag=1.6, kern=(0.3, 1.05, 2.0)),
    # today's broad kernel, but measured from each ball's entry time (counter lag = entry + D): fixes the queue
    # drift in hopper dumps without narrowing the window
    "entry_q_base_D1.7": entry_split(kern=(-0.9, 1.1, 3.8)),
    "entry_q_base_D1.4": entry_split(kern=(-0.6, 1.4, 4.1)),
    "entry_q_base_D2.0": entry_split(kern=(-1.2, 0.8, 3.5)),
    "entry_q_mid": entry_split(kern=(-0.4, 1.0, 3.0)),
    "const_lag1.7_base": entry_split(const_lag=1.7, kern=(-0.9, 1.1, 3.8)),
    # half today's counter-time kernel, half the narrow entry kernel
    "entry_q_mix0.5": entry_split(mix=0.5),
    "entry_q_mix0.3": entry_split(mix=0.3),
    "entry_q_mix0.7": entry_split(mix=0.7),
}


# ---------------------------------------------------------------- per match
DROP = 0.0  # --drop: share of each robot's detected launches removed (same seed in every arm): a recall probe


def _drop_launches(cap, k):
    if DROP <= 0:
        return cap
    for c in cap["calls"]:
        for ts in c["teams"]:
            if len(ts.shots):
                rng = np.random.default_rng(zlib.crc32(f"{k}|{ts.team}".encode()))
                keep = rng.random(len(ts.shots)) >= DROP
                ts.shots, ts.shot_w = ts.shots[keep], ts.shot_w[keep]
    return cap


def load_ctx(k):
    cap = _drop_launches(pickle.loads((DST / "capture" / f"{k}.pkl").read_bytes()), k)
    base = json.loads((DST / "ab" / "base" / k / "result.json").read_text())  # in-memory analyse, never re-written
    cache = pickle.loads((DST / "cache" / f"{k}.pkl").read_bytes())
    hf = {tuple(x.split("|")): np.array(v, dtype=np.float64).reshape(-1, 3)
          for x, v in json.loads((config.OUTPUT / k / "hub_feats.json").read_text()).items()}
    return cap, base, cache, hf


def run_arm(name, fn, k, cap, base, cache, hf):
    tl = TL(cap["auto_start"], cap["teleop_start"])
    a0 = cap["auto_start"] or 6.13
    ts_ = cap["teleop_start"] or a0 + 25.1
    t0, t1 = a0 - 1.0, ts_ + config.TELEOP_LEN + 8.0
    res = copy.deepcopy(base)
    log: list = []
    splits = {}
    for c in cap["calls"]:
        al = "red" if c["teams"][0].team in cap["red"] else "blue"
        ctx = {"tables": cache["views"], "hub_feats": hf, "al": al, "t0": t0, "t1": t1}
        saved = A.ATTR_MODEL
        A.ATTR_MODEL = c["model"]
        try:
            sp = fn(c["inc"], c["teams"], tl.match_time, log=log if c["logged"] else None, opr=c["opr"], ctx=ctx)
        finally:
            A.ATTR_MODEL = saved
        splits[(al, c["model"] is not None)] = sp
    res["fuel_log"] = log
    for al in ("red", "blue"):
        meta = res["teams"][f"_{al}"]
        scale = (meta["tba_total"] / meta["timeline_total"]) if (meta.get("tba_total") and meta.get("timeline_total")
                                                                 and meta.get("source") == "overlay") else 1.0
        sp = splits.get((al, True)) or splits.get((al, False))
        spv = splits.get((al, False)) or sp
        for tm in cap[al]:
            d = res["teams"][str(tm)]
            per = sp.get(tm, {})
            raw = sum(per.values())
            d["fuel"] = round(float(raw * scale), 1)
            d["fuel_by_period"] = {p: round(v * scale, 1) for p, v in sorted(per.items())}
            d["fuel_unscaled"] = round(raw, 1)
            d["fuel_video"] = round(sum(spv.get(tm, {}).values()) * scale, 1)
            d["points"] = round(raw * scale + d.get("auto_tower_points", 0) + d.get("endgame_tower_points", 0), 1)
    od = AB / name / k
    od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(json.dumps(res))
    return res


# ---------------------------------------------------------------- per-match metrics for the bootstrap
def per_match(out_root: Path) -> dict:
    """Per match: summed |err| (best / video-only), team-matches, top-scorer hits, alliances, and burst
    (balls, overlap*balls) exactly as score_scouting.score computes them."""
    s = score(out_root, keep_rows=True)
    pm: dict = {}
    for r in s["rows"]:
        d = pm.setdefault(r["key"], {"err": 0.0, "verr": 0.0, "n": 0, "top": 0, "vtop": 0, "al": 0,
                                     "bb": 0.0, "bo": 0.0, "rows": []})
        d["err"] += abs(r["video"] - r["ref"])
        d["verr"] += abs(r["vonly"] - r["ref"])
        d["n"] += 1
        d["rows"].append(r)
    for k, d in pm.items():
        g = {}
        for r in d["rows"]:
            g.setdefault(r["al"], []).append(r)
        for rr in g.values():
            ref = np.array([x["ref"] for x in rr])
            d["top"] += int(np.argmax([x["video"] for x in rr]) == np.argmax(ref))
            d["vtop"] += int(np.argmax([x["vonly"] for x in rr]) == np.argmax(ref))
            d["al"] += 1
        del d["rows"]
    # bursts (copy of score_scouting's burst block, kept per match)
    bm = scouting.by_match(scouting.load())
    for key, teams in sorted(bm.items()):
        rp = out_root / key / "result.json"
        if not rp.exists() or key not in pm:
            continue
        res = json.loads(rp.read_text())
        info = tba.match(key)
        to_mt = to_mt_fn(res["timeline"])
        a0 = res["timeline"].get("auto_start") or 6.13
        for al in ("red", "blue"):
            ts = info.teams(al)
            hub = info.hub(al) or {}
            if not hub.get("totalCount") or not all(teams.get(t) for t in ts):
                continue
            n = np.array([np.mean([r.n for r in teams[t]]) for t in ts])
            if n.sum() <= 0:
                continue
            log = [(mt, tm, b) for mt, tm, b in res.get("fuel_log") or [] if tm in ts]
            timed = {t: [r for r in teams[t] if r.time_valid] for t in ts}
            if not all(timed.values()):
                continue
            prev = 0
            for vt, c in res["timeline"]["counts"].get(al, []):
                if c <= prev:
                    continue
                kb = c - prev
                prev = c
                st_c = vt - a0
                w = np.array([np.mean([tri(r.shots - st_c).sum() for r in timed[t]]) for t in ts])
                if w.sum() <= 0:
                    continue
                tshare = w / w.sum()
                mt_log = to_mt(vt - 2.0)
                got = np.array([sum(b for m, tm, b in log if tm == t and abs(m - mt_log) < 0.06) for t in ts])
                if got.sum() <= 0:
                    continue
                vshare = got / got.sum()
                pm[key]["bb"] += kb
                pm[key]["bo"] += kb * float(np.minimum(vshare, tshare).sum())
    return {"score": {k: v for k, v in s.items() if k != "rows"}, "pm": pm}


def agg(pm, keys):
    e = sum(pm[k]["err"] for k in keys); v = sum(pm[k]["verr"] for k in keys); n = sum(pm[k]["n"] for k in keys)
    t = sum(pm[k]["top"] for k in keys); vt = sum(pm[k]["vtop"] for k in keys); a = sum(pm[k]["al"] for k in keys)
    bb = sum(pm[k]["bb"] for k in keys); bo = sum(pm[k]["bo"] for k in keys)
    return np.array([e / n, v / n, t / a, vt / a, bo / max(bb, 1e-9)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--drop", type=float, default=0.0)
    ap.add_argument("--dst", default=None, help="arm result folder (default data/exp_hubentry/ab)")
    a = ap.parse_args()
    global DROP, AB
    DROP = a.drop
    if a.dst:
        AB = Path(a.dst)
    arms = a.arms.split(",")
    keys = sorted(p.stem for p in (DST / "capture").glob("*.pkl"))
    print(len(keys), "captured matches", flush=True)
    for k in keys:  # noqa: B007
        cap, base, cache, hf = load_ctx(k)
        for name in arms:
            run_arm(name, ARMS[name], k, cap, base, cache, hf)
    results = {}
    for name in ["base"] + [x for x in arms if x != "base"]:
        results[name] = per_match(AB / name)
    ref = results["base"]["pm"]
    common = sorted(set(ref).intersection(*[set(r["pm"]) for r in results.values()]))
    rng = np.random.default_rng(0)
    boots = [rng.choice(len(common), len(common)) for _ in range(a.boot)]
    labels = ["best MAE", "video-only MAE", "top (best)", "top (video)", "burst"]
    table = {}
    b0 = agg(ref, common)
    for name, r in results.items():
        x = agg(r["pm"], common)
        d = []
        for bi in boots:
            ks = [common[i] for i in bi]
            d.append(agg(r["pm"], ks) - agg(ref, ks))
        d = np.array(d)
        nb = sum(1 for k in common if r["pm"][k]["err"] < ref[k]["err"] - 1e-9)
        nw = sum(1 for k in common if r["pm"][k]["err"] > ref[k]["err"] + 1e-9)
        table[name] = {"value": dict(zip(labels, x.tolist())), "diff": dict(zip(labels, (x - b0).tolist())),
                       "ci95": {lab: np.percentile(d[:, i], [2.5, 97.5]).tolist() for i, lab in enumerate(labels)},
                       "matches_better_worse": [nb, nw], "score": r["score"]}
        print(f"{name:16s} " + " | ".join(f"{lab} {x[i] * (100 if i >= 2 else 1):.2f} ({(x[i] - b0[i]) * (100 if i >= 2 else 1):+.2f} "
                                          f"[{np.percentile(d[:, i], 2.5) * (100 if i >= 2 else 1):+.2f}, "
                                          f"{np.percentile(d[:, i], 97.5) * (100 if i >= 2 else 1):+.2f}])"
                                          for i, lab in enumerate(labels)) + f" | better/worse {nb}/{nw}", flush=True)
    (DST / f"ab{a.tag}.json").write_text(json.dumps({"matches": common, "table": table}, indent=1))


if __name__ == "__main__":
    main()
