"""Re-run pipeline.calibrate's per-view fit on matches whose view is missing, and say why it was dropped."""
import sys
import numpy as np
from common import config, load, P
from autoscout import field as F, views
from autoscout.video import Video
for key, view in [("2026cancmp_qm114", "blue_end"), ("2026cancmp_qm52", "blue_end"), ("2026cagle_qm32", "red_end")]:
    r = load(key, "v1"); tl = r["timeline"]
    v = next(x for x in views.PRODUCTIONS["first"] if x.name == view)
    vid = Video(config.VIDEOS / f"{key}.mp4")
    tele = tl["teleop_start"]
    times = [float(t) for t in np.arange(tl["auto_start"] - 1, tele + config.TELEOP_LEN + 2, 4.0)]
    fr = [(t, vid.frame_at(t)) for t in times]
    fr = [(t, f) for t, f in fr if f is not None]
    crops = [v.crop(f) for _, f in fr]
    cam = F.calibrate_view_dynamic(crops, [t for t, _ in fr], F.load_tags(), v.mask_boxes(vid.width, vid.height))
    if cam is None:
        print(key, view, "calibrate_view_dynamic -> None")
        continue
    c = cam.at(cam.times[len(cam.times) // 2])
    print(key, view, f"rms {cam.rms:.2f} (pipeline keeps < 4.0) poses {len(cam.times)} tags {cam.n_tags} "
          f"f {cam.K[0][0]:.0f} k1 {cam.dist[0]:.3f} centre {np.round(c.center_world(), 2)}", flush=True)
