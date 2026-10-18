"""Download match videos for a team's events (or whole events) from TBA links."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import tba, video  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", type=int)
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--limit", type=int, default=0, help="max matches per event (0 = all)")
    a = ap.parse_args()
    events = a.events or [e["key"] for e in tba.team_events(a.team)]
    for ev in events:
        ms = tba.team_event_matches(a.team, ev) if a.team else tba.event_matches(ev)
        ms = [m for m in ms if m.youtube]
        if a.limit:
            ms = ms[: a.limit]
        for m in ms:
            try:
                p = video.download(m.key, m.youtube[0])
                print("ok", m.key, p.stat().st_size // 1_000_000, "MB", flush=True)
            except Exception as e:  # keep going; some videos are private/removed
                print("FAIL", m.key, e, flush=True)


if __name__ == "__main__":
    main()
