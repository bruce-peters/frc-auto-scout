"""Static main-camera fit (tags only) for every extract; data/exp_maincal/static_fits.json feeds the leave-one-out
event-prior variant of m12_eval ("event": median K, distortion and centre of the event's other matches, rotation
per window about that centre). Usage: m15_static_fits.py [prefix]
"""
import json
import sys

import numpy as np

from common import EXP, F, M, TAGS, keys, load_extract

out = {}
p = EXP / "static_fits.json"
if p.exists():
    out = json.loads(p.read_text())
for k in keys(sys.argv[1] if len(sys.argv) > 1 else ""):
    ex = load_extract(k)
    plain = ex["cams"].get("main")
    if plain is None:
        continue
    obs = [(t, d) for t, d in ex["tag_obs"] if d]
    c = M.static_candidates(obs, TAGS, ex["size"], plain=plain)
    if not c:
        continue
    tr = c[0][1]
    mid = tr.at(float(np.median(tr.times)))
    out[k] = {"K": tr.K, "dist": tr.dist, "centre": tr.static["centre"], "rvec": list(map(float, mid.rvec)),
              "size": list(ex["size"]), "rms": tr.rms, "n_tags": tr.n_tags, "start": tr.static.get("start")}
    print(k, round(tr.K[0][0]), tr.static["centre"], round(tr.rms, 3), tr.static.get("start"), flush=True)
p.write_text(json.dumps(out, indent=1), encoding="utf-8")


def loo_prior(key: str) -> dict | None:
    """Median K / distortion / centre of the same event's other matches (as field.load_calib_prior)."""
    ev = key.split("_")[0]
    ent = [e for k, e in out.items() if k.split("_")[0] == ev and k != key]
    if len(ent) < 3:
        return None
    f = float(np.median([e["K"][0][0] for e in ent]))
    ref = min(ent, key=lambda e: abs(e["K"][0][0] - f))
    return {"n": len(ent), "size": tuple(ent[0]["size"]), "f": f, "K": [[f, 0, ent[0]["K"][0][2]], [0, f, ent[0]["K"][1][2]], [0, 0, 1]],
            "dist": np.median([e["dist"] for e in ent], axis=0).tolist(),
            "centre": np.median([e["centre"] for e in ent], axis=0).tolist(), "rvec": ref["rvec"]}
