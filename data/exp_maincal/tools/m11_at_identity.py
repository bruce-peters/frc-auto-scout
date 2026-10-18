"""CameraTrack.at before / after the rvec-branch fix, on every stored camera of the extracts (all views):
max |rvec difference| over a 0.1 s grid. Must be exactly 0 wherever the stored rvecs are branch-consistent."""
import numpy as np
from common import F, keys, load_extract


def at_old(tr, t):
    ts = np.asarray(tr.times)
    if len(ts) == 1 or t <= ts[0]:
        return np.asarray(tr.rvecs[0]), np.asarray(tr.tvecs[0])
    if t >= ts[-1]:
        return np.asarray(tr.rvecs[-1]), np.asarray(tr.tvecs[-1])
    j = int(np.searchsorted(ts, t))
    a = (t - ts[j - 1]) / max(1e-9, ts[j] - ts[j - 1])
    return ((1 - a) * np.asarray(tr.rvecs[j - 1]) + a * np.asarray(tr.rvecs[j]),
            (1 - a) * np.asarray(tr.tvecs[j - 1]) + a * np.asarray(tr.tvecs[j]))


tot = {}
for k in keys():
    ex = load_extract(k)
    for v, cam in ex["cams"].items():
        if not hasattr(cam, "times"):
            continue
        d = 0.0
        for t in np.arange(cam.times[0] - 1, cam.times[-1] + 1, 0.1):
            r_old, t_old = at_old(cam, t)
            c = cam.at(float(t))
            d = max(d, float(np.abs(np.asarray(c.rvec) - r_old).max()), float(np.abs(np.asarray(c.tvec) - t_old).max()))
        ev = k.split("_")[0]
        tot.setdefault((ev, v), []).append(d)
        if d > 0 and not k.startswith("2026cc"):
            print("CHANGED", k, v, d)
for (ev, v), ds in sorted(tot.items()):
    print(f"{ev:12s} {v:9s} cams {len(ds):2d}  changed {sum(x > 0 for x in ds):2d}  max diff {max(ds):.3g}")
