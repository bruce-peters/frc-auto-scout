"""A/B of bumper-OCR settings on real re-scans (identity accuracy on held-out bumper reads).

For each match: one robot re-scan with models/robots_v2.pt into data/exp_ocr/scan/ (bumper crops kept as
JPEG), then each arm re-reads every bumper from scratch with its settings and assigns identities from the
even (odd) crops. Both arms are scored on the SAME yardstick: the union of both arms' reads of the
held-out crops, so an arm cannot win by grading itself with its own misreads.

  python scripts/ocr_ab.py 2026cagle_qm42 2026joh_qm39 ...
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from autoscout import config, ocr, pipeline  # noqa: E402
import rescan_robots  # noqa: E402

OUT = config.DATA / "exp_ocr"
ARMS = {
    "old (64 px, 12/seg, 700)": (64, dict(ocr_budget=700, ocr_per_s=0.0, ocr_all_cached=False)),
    "new (128 px, 1 per 2 s up to 40/seg, 2000)": (128, dict(ocr_budget=2000, ocr_per_s=0.5, ocr_all_cached=True)),
}


def load(key):
    with open(OUT / "scan" / key / "scan.pkl", "rb") as fh:
        return pickle.load(fh)


def main():
    keys = sys.argv[1:]
    tot = {a: [0.0, 0.0, 0, 0] for a in ARMS}  # agree, mass, reads, decisive crops
    for key in keys:
        if not (OUT / "scan" / key / "scan.pkl").exists():
            ocr.PREP_H = 64
            rescan_robots.rescan(key, str(config.MODELS / "robots_v2.pt"), OUT / "scan")
        caches = {}
        for arm, (h, o) in ARMS.items():
            ocr.PREP_H = h
            md = load(key)
            md.ocr_cache = {}
            pipeline.analyse(md, pipeline.Options(**o))
            caches[arm] = md.ocr_cache
        line = [key]
        for arm, (h, o) in ARMS.items():
            ocr.PREP_H = h
            agree = mass = 0.0
            for k in (0, 1):
                md = load(key)
                md.ocr_cache = dict(caches[arm])
                pipeline.analyse(md, pipeline.Options(ocr_holdout=k, **o))
                for d, row in zip(md.dets, md.det_rows):
                    cid = getattr(d, "crop_id", None)
                    if cid is None or cid % 2 != k or row[7] is None:
                        continue
                    reads = [r for a in ARMS for r in caches[a].get(cid, [])]
                    if not reads:
                        continue
                    teams = md.info.teams(d.alliance)
                    ev = ocr.team_evidence(reads, teams)
                    if ev.sum() <= 0:
                        continue
                    agree += ev[teams.index(row[7])] if row[7] in teams else 0.0
                    mass += ev.sum()
            n_reads = sum(1 for v in caches[arm].values() if v)
            tot[arm][0] += agree
            tot[arm][1] += mass
            tot[arm][2] += n_reads
            line.append(f"{100 * agree / max(mass, 1e-9):5.1f}% ({n_reads} reads)")
        print("  ".join(line), flush=True)
    print()
    for arm, (a, m, n, _) in tot.items():
        print(f"{arm:45s} held-out identity agreement {100 * a / max(m, 1e-9):.1f} %   non-empty reads {n}")
    (OUT / "result.json").write_text(json.dumps({a: {"agree": v[0], "mass": v[1], "reads": v[2]} for a, v in tot.items()},
                                                indent=1))


if __name__ == "__main__":
    main()
