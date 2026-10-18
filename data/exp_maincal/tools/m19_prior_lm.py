"""Event camera for the Chezy main view from 2026cc_qm3's tags + the two hand-picked centre-line landmarks
(k1 held at 0): data/exp_maincal/prior_lm.json, in field.load_calib_prior's format, for m12's event_lm variant."""
import json
import numpy as np
from common import EXP, F, M, MANUAL_PTS, TAGS, load_extract
ex = load_extract("2026cc_qm3")
plain = ex["cams"]["main"]
obs = [(t, d) for t, d in ex["tag_obs"] if d]
K, D, C = M.track_start(plain)
tr = M.fit_static(obs, TAGS, ex["size"], K, np.zeros(5), C, points=MANUAL_PTS, w_points=2.0, fix_k1=True)
mid = tr.at(float(np.median(tr.times)))
pr = {"event": "2026cc", "view": "main", "n": 1, "size": list(ex["size"]), "f": tr.K[0][0], "K": tr.K, "dist": tr.dist,
      "centre": tr.static["centre"], "rvec": list(map(float, mid.rvec)), "rms": tr.rms,
      "landmarks": np.hstack([MANUAL_PTS[0], MANUAL_PTS[1]]).tolist(), "source": "2026cc_qm3 tags + landmarks, k1 = 0"}
(EXP / "prior_lm.json").write_text(json.dumps(pr, indent=1), encoding="utf-8")
print(json.dumps({k: pr[k] for k in ("f", "centre", "rms")}))
