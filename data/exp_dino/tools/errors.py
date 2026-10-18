"""Where is the held-out identity error mass, and did the DINO probe know the answer?

For each segment of the held-out runs (k = 0, 1) with config --w: held-out OCR mass h, assigned team, the
probe's segment vote (argmax of mean per-crop log q). Error mass = sum(h) - h[assigned], split by
  mixed     h itself has mass on >= 2 teams (part of that is unavoidable: the ceiling sum(max h))
  dino_ok   the probe's vote = argmax h (the probe knew; the Viterbi did not follow)
  dino_bad  the probe voted for another team
  no_dino   no probe evidence (unknown teams)

  errors.py --events 2026cc --w 2 [--stride 3]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import eval_ident as E  # noqa: E402
from autoscout import pipeline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--w", type=float, default=2.0)
    ap.add_argument("--probe", default="box_inner_ocr")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    keys = sorted(p.parent.name for p in (ROOT / "output").glob("*/scan.pkl"))
    if a.events:
        keys = [k for k in keys if k.split("_")[0] in a.events]
    keys = keys[::a.stride]
    pipeline.DINO_PROBE_DIR = E.EXP / "probe" / a.probe
    tot = {"mass": 0.0, "agree": 0.0, "ceiling": 0.0, "mixed": 0.0, "dino_ok": 0.0, "dino_bad": 0.0, "no_dino": 0.0,
           "base_agree": 0.0, "fixed": 0.0, "broken": 0.0}
    for key in keys:
        with open(ROOT / "output" / key / "scan.pkl", "rb") as fh:
            md = pickle.load(fh)
        cp = E.OCRC / f"{key}.pkl"
        if cp.exists():
            with open(cp, "rb") as fh:
                md.ocr_cache.update(pickle.load(fh))
        lp = pipeline.load_dino_probe(key)
        info = md.info
        for k in (0, 1):
            base = E.identity_only(md, pipeline.Options(ocr_holdout=k))
            res = pipeline.analyse(md, pipeline.Options(ocr_holdout=k, w_dino=a.w))
            seg_crops: dict[int, list] = {}
            for d, r in zip(md.dets, md.det_rows):
                if d.crop_id is not None and r[8] is not None:
                    seg_crops.setdefault(int(r[8]), []).append(d.crop_id)
            for al in ("red", "blue"):
                teams = info.teams(al)
                for sid, s in res["identity"][al].items():
                    h = np.array(s["held"])
                    if s["team"] is None or h.sum() <= 0:
                        continue
                    ti = teams.index(s["team"])
                    bt = base[al][sid]["team"]
                    bi = teams.index(bt) if bt is not None else -1
                    tot["mass"] += h.sum()
                    tot["agree"] += h[ti]
                    tot["base_agree"] += h[bi] if bi >= 0 else 0.0
                    tot["ceiling"] += h.max()
                    if bi != ti:
                        tot["fixed" if h[ti] > (h[bi] if bi >= 0 else 0) else "broken"] += abs(h[ti] - (h[bi] if bi >= 0 else 0))
                    err = h.sum() - h[ti]
                    if err <= 1e-9:
                        continue
                    if (h > 0.15 * h.sum()).sum() >= 2:
                        tot["mixed"] += err
                        continue
                    rows = np.array([[lp[c].get(t, np.nan) for t in teams] for c in seg_crops.get(int(sid), []) if c in lp],
                                    dtype=float).reshape(-1, 3)
                    rows = rows[np.isfinite(rows).all(1)]
                    if not len(rows) or np.ptp(rows.mean(0)) < 1e-6:
                        tot["no_dino"] += err
                    elif rows.mean(0).argmax() == h.argmax():
                        tot["dino_ok"] += err
                        # is the right team held by a concurrent segment of the same alliance?
                        T = teams[int(h.argmax())]
                        conc = [o for oid, o in res["identity"][al].items() if oid != sid and o["team"] == T
                                and o["t0"] < s["t1"] and o["t1"] > s["t0"]]
                        if conc:
                            tot["conflict"] = tot.get("conflict", 0.0) + err
                            # does that concurrent segment also look like T (a duplicate track of one robot)?
                            dup = False
                            for o in conc:
                                oid = next(i for i, x in res["identity"][al].items() if x is o)
                                orow = np.array([[lp[c].get(t, np.nan) for t in teams]
                                                 for c in seg_crops.get(int(oid), []) if c in lp], dtype=float).reshape(-1, 3)
                                orow = orow[np.isfinite(orow).all(1)]
                                if len(orow) and teams[int(orow.mean(0).argmax())] == T:
                                    dup = True
                            if dup:
                                tot["conflict_dup"] = tot.get("conflict_dup", 0.0) + err
                    else:
                        tot["dino_bad"] += err
        print(f"{key}: agree {100 * tot['agree'] / tot['mass']:.1f}% (base {100 * tot['base_agree'] / tot['mass']:.1f}%) "
              f"ceiling {100 * tot['ceiling'] / tot['mass']:.1f}%", flush=True)
    m = tot["mass"]
    print({k: round(100 * v / m, 2) for k, v in tot.items()})


if __name__ == "__main__":
    main()
