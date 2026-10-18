"""Process many matches (skipping ones already done) and write reports."""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, pipeline, report, tba  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int, default=5026)
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--matches", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th match starting at i")
    ap.add_argument("--robot_model", default=None, help="robot detector weights (default: Options.robot_model)")
    a = ap.parse_args()
    keys = list(a.matches)
    for ev in a.events:
        keys += [m.key for m in tba.team_event_matches(a.team, ev) if m.youtube]
    if a.limit:
        keys = keys[: a.limit]
    si, sn = map(int, a.shard.split("/"))
    keys = keys[si::sn]
    opts = pipeline.Options(**({"robot_model": a.robot_model} if a.robot_model else {}))
    for k in keys:
        if not a.force and (config.OUTPUT / k / "result.json").exists():
            print("skip", k, flush=True)
            continue
        tic = time.time()
        try:
            res = pipeline.process_match(k, opts)
            report.write_html(res, config.OUTPUT / k)
            parts = []
            for al in ("red", "blue"):
                meta = res["teams"][f"_{al}"]
                parts.append(f"{al}: " + " ".join(f"{t}={v['fuel']:.0f}" for t, v in res["teams"].items()
                                                 if not str(t).startswith("_") and v["alliance"] == al)
                             + f" [{meta['source']} {meta['timeline_total']}/{meta['tba_total']}]")
            print(f"ok {k} {time.time() - tic:.0f}s | " + " | ".join(parts), flush=True)
        except Exception as e:
            print(f"FAIL {k}: {e!r}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
