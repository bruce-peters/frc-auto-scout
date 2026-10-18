"""Crop-strategy / label-policy comparison of the cross-match DINOv2 probe (no pipeline run).

Per event, leave-one-match-out: train the probe on the other matches (scripts/build_dino_probe.py
train_for) and score it on the held-out match's OCR-confident segments (the reference): which of the
alliance's 3 teams has the highest mean per-crop log q.

  compare.py --feats box_cls box_fg ctr_cls+ctr_mean --policy ocr --l2 1e-3 --events 2026cc --stride 3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np  # noqa: E402

import build_dino_probe as B  # noqa: E402

OUT = ROOT / "data" / "exp_dino" / "compare"


def score_match(m: dict, lq: np.ndarray) -> dict:
    """Reference = crops of confident segments (assigned team = clear OCR team)."""
    r = {"crop_n": 0, "crop_ok": 0, "seg_n": 0, "seg_cov": 0, "seg_ok": 0, "ll": 0.0,
         "view": {v: [0, 0] for v in range(3)}, "seg_main_n": 0, "seg_main_ok": 0}
    ref = np.where(m["conf"])[0]
    if not len(ref):
        return r
    a = m["al"][ref]
    q = np.stack([lq[i, 3 * aa:3 * aa + 3] for i, aa in zip(ref, a)])
    t = m["tidx"][ref]
    fin = np.isfinite(q).all(1)
    ok = fin & (np.nan_to_num(q, nan=-9).argmax(1) == t)
    r["crop_n"], r["crop_ok"] = int(fin.sum()), int(ok.sum())
    for v in range(3):
        mv = fin & (m["view"][ref] == v)
        r["view"][v] = [int(mv.sum()), int((ok & mv).sum())]
    for s in np.unique(m["seg"][ref]):
        ms = m["seg"][ref] == s
        r["seg_n"] += 1
        qs = q[ms & fin]
        if not len(qs):
            continue
        r["seg_cov"] += 1
        mean = qs.mean(0)
        good = int(mean.argmax() == t[ms][0])
        r["seg_ok"] += good
        r["ll"] += float(mean[t[ms][0]] - np.log(np.exp(mean).sum()))
        if (m["view"][ref][ms] == 0).mean() > 0.5:  # mostly the high main camera
            r["seg_main_n"] += 1
            r["seg_main_ok"] += good
    return r


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=[])
    ap.add_argument("--policy", nargs="+", default=["ocr"])
    ap.add_argument("--l2", nargs="+", type=float, default=[1e-3])
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_h", type=int, default=0, help="train only on crops at least this tall")
    ap.add_argument("--tag", default="cmp")
    ap.add_argument("--from_probe", nargs="*", default=[], help="score existing data/exp_dino/probe/<name> files")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    evm = B.event_matches()
    rows = []
    for name in a.from_probe:  # no training: score the stored leave-one-match-out probe outputs
        for ev, keys in evm.items():
            if a.events and ev not in a.events:
                continue
            tot = {"seg_ok": 0, "seg_cov": 0, "seg_n": 0, "crop_ok": 0, "crop_n": 0}
            for k in keys:
                m = B.load_match(k, ["box_cls"])
                P = np.load(B.PROBE / name / f"{k}.npz")
                pos = {int(c): i for i, c in enumerate(P["ids"])}
                lq = np.stack([P["logq"][pos[int(c)]] if int(c) in pos else np.full(6, np.nan) for c in m["ids"]])
                r = score_match(m, lq.astype(np.float32))
                for kk in tot:
                    tot[kk] += r[kk]
            print(f"{name:22s} {ev:10s} seg {tot['seg_ok'] / max(1, tot['seg_cov']):.3f} (cov {tot['seg_cov'] / max(1, tot['seg_n']):.2f}, "
                  f"n {tot['seg_n']})  crop {tot['crop_ok'] / max(1, tot['crop_n']):.3f} (n {tot['crop_n']})", flush=True)
    if a.from_probe:
        return
    for feat in a.feats:
        for ev, keys in evm.items():
            if a.events and ev not in a.events:
                continue
            data = [m for m in (B.load_match(k, feat.split("+")) for k in keys) if m is not None]
            for pol in a.policy:
                for l2 in a.l2:
                    tic = time.time()
                    tot: dict = {}
                    for i in range(0, len(data), a.stride):
                        model, classes = B.train_for(data[:i] + data[i + 1:], pol, l2, dev, a.min_h)
                        r = score_match(data[i], B.probe_logq(model, classes, data[i], dev))
                        for k, v in r.items():
                            if k == "view":
                                tv = tot.setdefault("view", {vv: [0, 0] for vv in range(3)})
                                for vv in range(3):
                                    tv[vv][0] += v[vv][0]
                                    tv[vv][1] += v[vv][1]
                            else:
                                tot[k] = tot.get(k, 0) + v
                    row = {"feat": feat, "event": ev, "policy": pol, "l2": l2, **tot,
                           "seg_acc": tot["seg_ok"] / max(1, tot["seg_cov"]),
                           "seg_cov_rate": tot["seg_cov"] / max(1, tot["seg_n"]),
                           "crop_acc": tot["crop_ok"] / max(1, tot["crop_n"]),
                           "main_seg_acc": tot["seg_main_ok"] / max(1, tot["seg_main_n"]),
                           "mean_ll": tot["ll"] / max(1, tot["seg_cov"])}
                    vacc = " ".join(f"{['main', 'red', 'blue'][v]} {tot['view'][v][1] / max(1, tot['view'][v][0]):.3f}"
                                    for v in range(3))
                    print(f"{feat:22s} {ev:10s} {pol:8s} l2={l2:g}  seg {row['seg_acc']:.3f} (cov {row['seg_cov_rate']:.2f}, "
                          f"n {tot['seg_n']})  main-seg {row['main_seg_acc']:.3f} (n {tot['seg_main_n']})  crop {row['crop_acc']:.3f} "
                          f"[{vacc}]  ll {row['mean_ll']:.3f}  ({time.time() - tic:.0f}s)", flush=True)
                    rows.append(row)
            del data
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.tag}.json").write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()
