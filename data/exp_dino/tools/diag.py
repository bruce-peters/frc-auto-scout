"""Where does the DINO probe change identities, and is it right? One match, held-out split k.

  diag.py 2026cagle_sf1m1 [--w 4] [--probe box_inner_ocr]
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
    ap.add_argument("key")
    ap.add_argument("--w", type=float, default=4.0)
    ap.add_argument("--probe", default="box_inner_ocr")
    a = ap.parse_args()
    with open(ROOT / "output" / a.key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    cp = E.OCRC / f"{a.key}.pkl"
    if cp.exists():
        with open(cp, "rb") as fh:
            md.ocr_cache.update(pickle.load(fh))
    pipeline.DINO_PROBE_DIR = E.EXP / "probe" / a.probe
    lp = pipeline.load_dino_probe(a.key)
    info = md.info
    for k in (0, 1):
        base = E.identity_only(md, pipeline.Options(ocr_holdout=k))
        dino =E.identity_only(md, pipeline.Options(ocr_holdout=k, w_dino=a.w))
        for al in ("red", "blue"):
            teams = info.teams(al)
            print(f"--- k={k} {al} {teams}")
            for sid, s in base[al].items():
                d = dino[al][sid]
                if s["team"] == d["team"]:
                    continue
                h = np.array(s["held"])
                print(f"  seg {sid:>4} slot {s['slot']} {s['t0']:6.1f}-{s['t1']:6.1f}  ocr {s['ocr']}  held {h.round(1).tolist()}  "
                      f"base {s['team']} -> dino {d['team']}  {'BETTER' if h.sum() and h[teams.index(d['team'])] > h[teams.index(s['team'])] else ('WORSE' if h.sum() else 'no ref')}")
    # per-team probe sanity: mean q of each alliance team over crops labelled (confidently) as each team
    L = np.load(E.LAB / f"{a.key}.npz")
    pos = {int(c): i for i, c in enumerate(L["ids"])}
    for al_i, al in enumerate(("red", "blue")):
        teams = info.teams(al)
        print(f"probe confusion ({al}): rows = assigned team of confident segments, cols = mean q")
        for t in teams:
            rows = []
            for cid, row in lp.items():
                i = pos.get(cid)
                if i is None or L["team"][i] != t or L["al"][i] != al_i:
                    continue
                o = L["ocr"][i]
                if o.sum() < 2 or o.max() < 0.8 * o.sum():
                    continue
                rows.append([np.exp(row[tt]) for tt in teams])
            if rows:
                print(f"  {t:>5}: n={len(rows):4d} " + " ".join(f"{v:.2f}" for v in np.nanmean(rows, 0)))


if __name__ == "__main__":
    main()
