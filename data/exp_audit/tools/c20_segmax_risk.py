"""Cycle 20a: SEG_MAX risk check on current output/ (SEG_MAX=10): label flips between adjacent pieces of
one tracker segment (same slot, contiguous: gap < 0.25 s) - with reads on either side vs on neither."""
import numpy as np
from collections import Counter
from common import load, match_keys, tba
c = Counter()
for k in match_keys("v2"):
    r = load(k, "v2")
    for al in ("red", "blue"):
        segs = sorted(((e["slot"], e["t0"], e["t1"], e["team"], sum(e["ocr"])) for e in r["identity"][al].values()))
        for a, b in zip(segs[:-1], segs[1:]):
            if a[0] != b[0] or not (0 <= b[1] - a[2] < 0.25):
                continue
            reads = "reads" if (a[4] > 0 or b[4] > 0) else "no reads"
            c[(reads, "flip" if a[3] != b[3] else "keep")] += 1
for reads in ("reads", "no reads"):
    f, kk = c[(reads, "flip")], c[(reads, "keep")]
    print(f"adjacent pieces with {reads:8s}: {f + kk:6d}, label flips {f:5d} ({100 * f / max(f + kk, 1):.1f} %)")
