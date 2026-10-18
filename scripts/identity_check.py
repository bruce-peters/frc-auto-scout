"""Identity consistency: share of bumper-OCR evidence that agrees with each segment's assigned team."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, tba  # noqa: E402


def main():
    tot_agree = tot_mass = 0.0
    for f in sorted(config.OUTPUT.glob("*/result.json")):
        r = json.loads(f.read_text())
        info = tba.match(r["match"])
        line = []
        for al in ("red", "blue"):
            teams = info.teams(al)
            agree = mass = 0.0
            for s in r["identity"][al].values():
                if s["team"] is None:
                    continue
                ti = teams.index(s["team"])
                agree += s["ocr"][ti]
                mass += sum(s["ocr"])
            tot_agree += agree
            tot_mass += mass
            line.append(f"{al} {100 * agree / mass:5.1f}% of {mass:5.1f}" if mass else f"{al}   n/a")
        print(f"{r['match']:22s} " + "  ".join(line))
    if tot_mass:
        print(f"\noverall OCR agreement with assigned identities: {100 * tot_agree / tot_mass:.1f}%")


if __name__ == "__main__":
    main()
