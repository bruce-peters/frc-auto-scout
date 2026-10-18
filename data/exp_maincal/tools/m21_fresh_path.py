"""Smoke test of the fresh-scan path without a rescan: pipeline.calibrate on the video (event set, fallbacks) as
process_match calls it, then the post-scan step (accept_main_cam) on the saved detections of a scan copy.
Uses a scratch cache (data/exp_maincal/cache) so data/cache is not touched.
Usage: m21_fresh_path.py match...
"""
import pickle
import sys
import time

from common import EXP, F, M, P

F.config.CACHE = EXP / "cache"
F.config.CACHE.mkdir(exist_ok=True)
from autoscout import config  # noqa: E402
from autoscout.video import Video  # noqa: E402

for key in sys.argv[1:]:
    tic = time.time()
    with open(EXP / "before" / key / "scan.pkl", "rb") as fh:
        md = pickle.load(fh)
    tl = md.timeline
    tele = tl.teleop_start or (tl.auto_start + P.FIRST_TELEOP_GAP)
    recover = {}
    video = Video(config.VIDEOS / f"{key}.mp4")
    cams = P.calibrate(video, md.prod, tl.auto_start - 1, tele + config.TELEOP_LEN + 2, event=md.info.event_key,
                       match=key, fallbacks=recover)
    video.close()
    t_cal = time.time() - tic
    main = cams.get("main")
    sc = M.centre_scatter(main) if main is not None else None
    print(f"{key}: calibrate {t_cal:.0f}s, views {sorted(cams)}, main scatter {sc:.2f} m, recover keys {sorted(recover)}")
    if "main" in recover:
        cands, trusted = recover.pop("main")
        md.cams = dict(md.cams, main=main)  # the fresh plain fit, as the scan would hold it
        r = P.accept_main_cam(md, cands, prefer="event_prior", trusted=trusted)
        print(f"   candidates {[c[0] for c in cands]} trusted {trusted} -> accepted {r['accepted']} flag {r['flag']}; "
              f"current off {r['current']['off']} | " + " | ".join(f"{c['how']} off {c['off']} dxv {c['d_xview']} pass {c['pass']}"
                                                              for c in r["candidates"]))
    print(f"   prior file: {sorted((F._read_json(F.calib_prior_path(md.info.event_key, 'main')) or {}).get('matches', {}))}"
          f" ({time.time() - tic:.0f}s)", flush=True)
