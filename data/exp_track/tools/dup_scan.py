"""Per video: is it 30p content in a 60 fps file (every picture shown twice)? Mean |difference| of consecutive
grey frames (main view) over 240 frames at 4 points of the match; duplicated pictures make every other difference
tiny. ratio = mean(smaller phase) / mean(larger phase): ~0.1 for duplicated frames, ~1 for true 60p.
dup_share = share of consecutive frames whose difference is < 0.35 x the median of the larger half of the
differences around it (+-15 frames): ~0.5 for 30p shown twice (also with an irregular cadence), ~0 for 60p."""
import json, sys
from pathlib import Path
import cv2, numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, views
from autoscout.video import Video, video_path
out = {}
keys = sorted(p.parent.name for p in config.OUTPUT.glob("*/scan.pkl"))
for key in keys:
    v = Video(video_path(key))
    rs, shares, raw = [], [], []
    for t0 in (40, 80, 120, 160):
        start = int(t0 * v.fps)
        prev, d = None, []
        for idx, f in v.iter_frames(start, start + 121):
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[: int(0.65 * f.shape[0])].astype(np.int16)
            if prev is not None:
                d.append(float(np.abs(g - prev).mean()))
            prev = g
        d = np.array(d)
        a, b = d[0::2].mean(), d[1::2].mean()
        rs.append(min(a, b) / max(a, b, 1e-9))
        tiny = []
        for i in range(len(d)):
            w = d[max(0, i - 15):i + 16]
            big = np.median(np.sort(w)[len(w) // 2:])
            tiny.append(d[i] < 0.35 * big)
        shares.append(float(np.mean(tiny)))
        raw.append([round(float(x), 3) for x in d])
    prod = views.detect_production(v.frame_at(min(v.duration - 1, 60)))
    out[key] = {"fps": round(v.fps, 3), "prod": prod, "ratio": [round(float(x), 3) for x in rs],
                "dup_share": [round(x, 3) for x in shares], "raw": raw}
    v.close()
    print(key, {k_: v_ for k_, v_ in out[key].items() if k_ != "raw"}, flush=True)
(config.DATA / "exp_track" / "dup_frames.json").write_text(json.dumps(out, indent=1))
from collections import Counter
c = Counter((x["prod"], x["fps"], x["dup"]) for x in out.values())
print("\n(production, fps, duplicated frames): matches")
for k_, n in sorted(c.items()):
    print("  ", k_, n)
