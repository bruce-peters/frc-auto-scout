import json, sys
from common import A, AUDIT, TL, load, match_keys, resplit, tba, team_signals
import c6e_auto_variants as V  # runs the variant table once (prints); then reuse make()
src = "v2"
out = {"fuel": {}}
for mode in ("as_is", "off"):
    out["fuel"][mode] = {}
    for k in V.R:
        A.split_increments = V.make(mode, TL(V.R[k]))
        out["fuel"][mode][k] = {str(t): f for t, f in resplit(V.R[k], V.INFO[k], signals=V.SIG[k]).items()}
    A.split_increments = V.ORIG
json.dump(out, open(AUDIT / "c6f_auto.json", "w"))
