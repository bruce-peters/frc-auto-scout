"""CLI:  python -m autoscout <match_key> [<match_key> ...] [--debug]
        python -m autoscout --team 5026 --event 2026casnv
        python -m autoscout 2026cc_qm54 --reanalyse     # from the cached scan: seconds, no video or GPU
"""
from __future__ import annotations

import argparse

from . import config, pipeline, report, tba


def main():
    ap = argparse.ArgumentParser(prog="autoscout")
    ap.add_argument("matches", nargs="*")
    ap.add_argument("--team", type=int)
    ap.add_argument("--event")
    ap.add_argument("--model", default=None, help="robot detector weights")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--reanalyse", action="store_true",
                    help="re-run tracking, identity and fuel credit from output/<match>/scan.pkl (no video needed)")
    a = ap.parse_args()
    keys = list(a.matches)
    if a.event:
        ms = tba.team_event_matches(a.team, a.event) if a.team else tba.event_matches(a.event)
        keys += [m.key for m in ms if m.youtube]
    opts = pipeline.Options(debug=a.debug)
    if a.model:
        opts.robot_model = a.model
    for k in keys:
        try:
            res = pipeline.reanalyse(k, opts) if a.reanalyse else pipeline.process_match(k, opts)
        except Exception as e:  # keep going over a batch
            print(f"{k}: FAILED {e!r}")
            if a.debug:
                raise
            continue
        report.write_html(res, config.OUTPUT / k)
        print(f"{k} ({res['runtime_s']} s)")
        for al in ("red", "blue"):
            meta = res["teams"][f"_{al}"]
            parts = [f"{t}:{v['fuel']:.0f}" for t, v in res["teams"].items()
                     if not str(t).startswith("_") and v["alliance"] == al]
            print(f"  {al:4s} {'  '.join(parts)}   [{meta['source']}, total {meta['timeline_total']}, TBA {meta['tba_total']}]")


if __name__ == "__main__":
    main()
