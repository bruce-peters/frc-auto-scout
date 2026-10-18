"""Re-read the score overlay for selected matches and re-analyse them from their cached scans.

Use after changing overlay ROIs/parsing: only the overlay pass (a few seconds of OCR per
video second sampled) is redone, not the full video scan.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from autoscout import config, overlay, pipeline, report, tba  # noqa: E402
from autoscout.video import Video, video_path  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--playoffs", action="store_true", help="all non-qualification matches")
    ap.add_argument("--events", nargs="*", default=[])
    ap.add_argument("--matches", nargs="*", default=[])
    a = ap.parse_args()
    keys = []
    for f in sorted(config.OUTPUT.glob("*/scan.pkl")):
        k = f.parent.name
        info = tba.match(k)
        if k in a.matches or info.event_key in a.events or (a.playoffs and info.comp_level != "qm"):
            keys.append(k)
    for k in keys:
        v = Video(video_path(k))
        layout, rows = overlay.scan(v, hz=3.0)
        (config.CACHE / f"overlay_{k}.json").write_text(json.dumps({"layout": layout, "rows": rows}, default=str))
        tl = overlay.build_timeline(layout, rows)
        pk = config.OUTPUT / k / "scan.pkl"
        with open(pk, "rb") as fh:
            md = pickle.load(fh)
        if tl.auto_start is not None:
            md.timeline = tl
        with open(pk, "wb") as fh:
            pickle.dump(md, fh, protocol=pickle.HIGHEST_PROTOCOL)
        res = pipeline.reanalyse(k)
        res = json.loads((config.OUTPUT / k / "result.json").read_text())
        report.write_html(res, config.OUTPUT / k)
        info = tba.match(k)
        tot = {al: (res["teams"][f"_{al}"]["timeline_total"], info.hub(al).get("totalCount")) for al in ("red", "blue")}
        print(k, layout, tot, flush=True)


if __name__ == "__main__":
    main()
