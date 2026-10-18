"""Why did the end-camera calibration fail? Tags per sampled frame in the missing view (+ one crop each)."""
import json, sys
import cv2
import numpy as np
from common import AUDIT, config, load, P
from autoscout import field as F, views
from autoscout.video import Video
CASES = [("2026cagle_qm32", "red_end"), ("2026cagle_sf13m1", "red_end"), ("2026cancmp_qm114", "blue_end"),
         ("2026cancmp_qm52", "blue_end"), ("2026casnv_qm7", "blue_end"), ("2026cc_qm15", "blue_end"), ("2026joh_qm14", "blue_end")]
ims = []
out = {}
for key, view in CASES:
    r = load(key, "v1"); tl = r["timeline"]
    prod = "chezy" if tl["layout"] == "chezy" else "first"
    v = next(x for x in views.PRODUCTIONS[prod] if x.name == view)
    vid = Video(config.VIDEOS / f"{key}.mp4")
    a0 = tl["auto_start"]; t1 = (tl["teleop_start"] or a0 + 25) + 142
    n = []
    ids = {}
    for t in np.arange(a0 - 1, t1, 8.0):
        f = vid.frame_at(float(t))
        if f is None:
            continue
        d = F.detect_tags(v.crop(f))
        n.append(len(d))
        for i in d:
            ids[i] = ids.get(i, 0) + 1
    f = vid.frame_at(a0 + 60)
    c = cv2.resize(v.crop(f), (640, 230))
    cv2.putText(c, f"{key} {view}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    ims.append(c)
    out[key] = {"view": view, "tags_per_frame": np.bincount(n).tolist(), "ids": ids}
    print(key, view, "frames by #tags", np.bincount(n).tolist(), "ids", ids, flush=True)
while len(ims) % 2:
    ims.append(np.zeros_like(ims[0]))
cv2.imwrite(str(AUDIT / "img" / "uncalibrated_views.jpg"), np.vstack([np.hstack(ims[i:i + 2]) for i in range(0, len(ims), 2)]),
            [cv2.IMWRITE_JPEG_QUALITY, 85])
json.dump(out, open(AUDIT / "c3h_whyfail.json", "w"), indent=1)
