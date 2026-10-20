"""30p-in-60 check on a 60 Hz blob log: blob counts on even / odd frames, repeated blob positions between
consecutive frames, and shot tracks from all frames vs one frame of each duplicate pair."""
import gzip, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, fuel, pipeline as P, tracks60
for key in sys.argv[1:]:
    log = pickle.load(gzip.open(tracks60.log_path(key), "rb"))
    md = pickle.load(open(config.OUTPUT / key / "scan.pkl", "rb"))
    rows = log["blob_log"]["main"]
    n = np.array([len(r[1]) for r in rows])
    # repeated positions: share of a frame's blobs with a blob within 1.5 px in the previous frame
    rep = []
    for a, b in zip(rows[:-1], rows[1:]):
        A, B = a[1][:, :2], b[1][:, :2]
        if len(A) and len(B):
            d = np.sqrt(((B[:, None] - A[None]) ** 2).sum(-1)).min(1)
            rep.append(float((d < 1.5).mean()))
        else:
            rep.append(np.nan)
    rep = np.array(rep)
    def shots(sel):
        tot = 0
        for vn, rr in log["blob_log"].items():
            rr = [r for i, r in enumerate(rr) if sel(i)]
            tr = fuel.FlightTracker(gate=45.0 * (1 if len(rr) == len(log["blob_log"][vn]) else 2) ** 0.7)
            for r in rr:
                tr.update(r[0], [fuel.Blob(*map(float, x)) for x in r[1]])
            cam_t = md.cams.get(vn)
            for t in tr.flush():
                cam = cam_t.at(t.pts[0][0]) if cam_t else None
                tot += fuel.is_shot(t, P._px_per_m(cam, t.pts[0][1], t.pts[0][2]))
        return tot
    print(f"{key}: main-view blobs per frame even {n[0::2].mean():.1f} / odd {n[1::2].mean():.1f}; blobs repeated "
          f"(< 1.5 px) from the previous frame: frames 2k->2k+1 {np.nanmean(rep[0::2]):.2f}, 2k+1->2k+2 {np.nanmean(rep[1::2]):.2f}")
    print(f"   shot tracks (all views): all frames (60 Hz) {shots(lambda i: True)}, even frames only {shots(lambda i: i % 2 == 0)}, "
          f"odd frames only {shots(lambda i: i % 2 == 1)}")
