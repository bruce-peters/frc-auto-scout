"""Fit / evaluate the vision-only hub counter against the overlay counter.

For every processed match with an overlay timeline, per-second hub-activity features are
regressed (non-negative) onto the overlay's per-second increments. Evaluation is
leave-one-match-out on whole-match alliance totals vs TBA (active-hub periods only), which is
exactly the "alliance fuel within 15 %" acceptance test, done without any overlay or TBA input
at prediction time except the hub schedule.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import attribution, config, tba, vision_count as vc  # noqa: E402

LAG = 2.5  # s, entry activity -> counter increment
SMOOTH = 3


def active_mask(info: tba.MatchInfo, alliance: str, t_bins: np.ndarray, to_match_time) -> np.ndarray:
    """1 where the alliance hub counts fuel (game-manual shift schedule)."""
    inactive = tba.inactive_periods(info)[alliance]
    m = np.zeros(len(t_bins))
    for i, t in enumerate(t_bins):
        mt = to_match_time(t - LAG)
        if mt is None or mt < -1 or mt > 163:
            continue
        m[i] = 0.0 if attribution.period_of(mt) in inactive else 1.0
    return m


def collect():
    data = []
    for f in sorted(config.OUTPUT.glob("*/result.json")):
        r = json.loads(f.read_text())
        key = r["match"]
        feats = vc.load_feats(key)
        if not feats:
            continue
        tl = r["timeline"]
        if tl.get("auto_start") is None:
            continue
        a0, ts = tl["auto_start"], tl.get("teleop_start")

        def to_mt(t, a0=a0, ts=ts):
            return (20 + t - ts) if ts and t >= ts else t - a0

        info = tba.match(key)
        for al in ("red", "blue"):
            t, X = vc.binned(feats, al, a0 - 1, (ts or a0 + 25) + 146)
            X = vc.smooth(X, SMOOTH)
            counts = tl.get("counts", {}).get(al) or []
            y = vc.target_from_counts(counts, t, LAG) if counts and r["teams"][f"_{al}"]["source"] == "overlay" else None
            act = active_mask(info, al, t, to_mt)
            data.append({"match": key, "alliance": al, "t": t, "X": X, "y": y, "active": act,
                         "tba": info.hub(al).get("totalCount"), "event": info.event_key})
    return data


def main():
    data = collect()
    train = [d for d in data if d["y"] is not None]
    print(f"{len(data)} alliance-matches with features, {len(train)} with overlay targets")
    if len(train) < 4:
        return
    errs = []
    for i, d in enumerate(data):
        if d["tba"] is None:
            continue
        # leave this match (both alliances) out
        tr = [(x["X"] * x["active"][:, None], x["y"] * x["active"]) for x in train if x["match"] != d["match"]]
        w = vc.fit(tr)
        pred = float(np.clip(vc.predict(w, d["X"]), 0, None) @ d["active"])
        e = 100 * (pred - d["tba"]) / max(d["tba"], 1)
        errs.append((d["match"], d["alliance"], d["tba"], pred, e))
    for m, al, tb, p, e in errs:
        print(f"{m:22s} {al:4s} TBA {tb:4d}  vision {p:6.0f}  err {e:6.1f}%")
    ae = np.abs([e[-1] for e in errs if e[2] >= 20])
    print(f"\nLOO vision-only (alliances with >=20 fuel): n={len(ae)} mean |err| {ae.mean():.1f}%  "
          f"median {np.median(ae):.1f}%  within 15%: {np.mean(ae <= 15) * 100:.0f}%")
    w = vc.fit([(x["X"] * x["active"][:, None], x["y"] * x["active"]) for x in train])
    vc.save(w, {"lag": LAG, "smooth": SMOOTH, "n_train": len(train)})
    print("weights:", {n: round(v, 3) for n, v in zip(vc.feature_names() + ["bias"], w) if v > 1e-4})
    (config.OUTPUT / "vision_counter_eval.json").write_text(json.dumps(errs, default=float, indent=1))


if __name__ == "__main__":
    main()
