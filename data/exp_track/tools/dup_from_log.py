"""Duplicated-picture test from a 60 Hz blob log alone: share of a frame's main-view blobs that reappear within
1.5 px in the next frame, for the two frame phases. Duplicated pictures: one phase is high (the copy)."""
import gzip, json, pickle, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from autoscout import config, tracks60
res = {}
for p in sorted(Path(config.BLOBS60).glob("*.pkl.gz")):
    key = p.name.split(".")[0]
    rows = pickle.load(gzip.open(p, "rb"))["blob_log"]["main"]
    rep = []
    for a, b in zip(rows[:-1], rows[1:]):
        A, B = a[1][:, :2], b[1][:, :2]
        rep.append(float((np.sqrt(((B[:, None] - A[None]) ** 2).sum(-1)).min(1) < 1.5).mean()) if len(A) and len(B) else np.nan)
    rep = np.array(rep)
    res[key] = [round(float(np.nanmean(rep[0::2])), 3), round(float(np.nanmean(rep[1::2])), 3)]
    print(key, res[key], flush=True)
(config.DATA / "exp_track" / "dup_from_log.json").write_text(json.dumps(res, indent=1))
