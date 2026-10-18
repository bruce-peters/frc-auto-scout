"""Held-out identity accuracy: assign identities from half of the bumper reads, score on the rest.

Accuracy = share of held-out OCR evidence (weighted by segment) that agrees with the identity
assigned without it. Uses cached scans, so it is fast.

  python scripts/identity_holdout.py                          # output/ -> output/identity_holdout.json
  python scripts/identity_holdout.py --out data/exp_robots_v2/v2 [--matches k1 k2 ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoscout import config, pipeline, tba  # noqa: E402


def main(out: Path | str | None = None, matches: set[str] | None = None, **kw):
    """`out`: directory of <match>/scan.pkl (default output/); `matches`: only these keys.
    The summary json is written into `out` unless options or a match subset are given."""
    root = Path(out) if out else config.OUTPUT
    files = [f for f in sorted(root.glob("*/scan.pkl")) if matches is None or f.parent.name in matches]
    agree = mass = 0.0
    per_match = {}
    for f in files:
        key = f.parent.name
        info = tba.match(key)
        a_m = m_m = 0.0
        for k in (0, 1):
            extra = {a: b for a, b in kw.items() if a not in ("quiet", "use_galleries")}
            if kw.get("use_galleries"):
                import build_galleries
                extra["galleries"] = build_galleries.galleries_for(key, info.event_key)
            opts = pipeline.Options(ocr_holdout=k, out_dir=root, **extra)
            import pickle
            with open(f, "rb") as fh:
                md = pickle.load(fh)
            res = pipeline.analyse(md, opts)
            for al in ("red", "blue"):
                teams = info.teams(al)
                for s in res["identity"][al].values():
                    if s["team"] is None:
                        continue
                    h = s["held"]
                    a_m += h[teams.index(s["team"])]
                    m_m += sum(h)
        agree += a_m
        mass += m_m
        per_match[key] = (a_m, m_m)
        if not kw.get("quiet", False):
            print(f"{key:22s} held-out agreement {100 * a_m / max(m_m, 1e-9):5.1f}%  (evidence {m_m:.0f})", flush=True)
    print(f"overall held-out identity agreement: {100 * agree / max(mass, 1e-9):.1f}%  "
          f"({len(files)} matches, {root})  {kw}", flush=True)
    if not (set(kw) - {"quiet"}) and matches is None:
        (root / "identity_holdout.json").write_text(json.dumps(
            {"agreement": agree / max(mass, 1e-9), "evidence": mass, "matches": len(files),
             "per_match": per_match}))
    return agree / max(mass, 1e-9)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="directory of <match>/scan.pkl (default output/)")
    ap.add_argument("--matches", nargs="+", default=None, help="only these match keys")
    a = ap.parse_args()
    main(out=a.out, matches=set(a.matches) if a.matches else None)
