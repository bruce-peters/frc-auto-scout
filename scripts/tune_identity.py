"""Grid-search identity weights against held-out OCR agreement (uses cached scans)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import identity_holdout  # noqa: E402


def main():
    grid = {"w_app": [0.0, 3.0, 8.0, 20.0], "switch_cost": [0.3, 1.0, 3.0]}
    best = None
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(grid, vals))
        acc = identity_holdout.main(quiet=True, **kw)
        if best is None or acc > best[0]:
            best = (acc, kw)
    print("best", best)


if __name__ == "__main__":
    main()
