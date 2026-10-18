"""Re-run analysis on every cached scan with the current code, then rebuild all reports.

1. pipeline.reanalyse() for every output/<match>/scan.pkl (tracking, identity, attribution)
2. per-match report.html, output/index.html, team summary
3. vision-only counter fit (leave-one-match-out), per-team check against human scouting
   (scripts/score_scouting.py) and scripts/evaluate.py -> RESULTS.md
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, pipeline, report  # noqa: E402

PY = sys.executable
HERE = Path(__file__).resolve().parent


def main():
    tic = time.time()
    keys = sorted(p.parent.name for p in config.OUTPUT.glob("*/scan.pkl"))
    for k in keys:
        try:
            res = pipeline.reanalyse(k)
            res = json.loads((config.OUTPUT / k / "result.json").read_text())
            report.write_html(res, config.OUTPUT / k)
            print("reanalysed", k, flush=True)
        except Exception as e:
            print("FAIL", k, repr(e), flush=True)
    report.write_index()
    for script, args in (("fit_vision_counter.py", []), ("identity_holdout.py", []),
                         ("team_summary.py", ["--team", "5026"]),
                         ("score_scouting.py", ["--write", "--label", "current results"]), ("evaluate.py", [])):
        print(f"--- {script}", flush=True)
        subprocess.run([PY, str(HERE / script), *args], check=False)
    print(f"done in {time.time() - tic:.0f}s")


if __name__ == "__main__":
    main()
