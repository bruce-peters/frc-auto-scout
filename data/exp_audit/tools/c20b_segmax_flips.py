"""SEG_MAX pieces (consecutive same-slot pieces of equal length 5-10 s, gap < 0.25 s): label flips by whether
the pieces carry reads, and whether a flip agrees with the reads of the piece it lands on."""
import numpy as np
from collections import Counter
from common import load, match_keys, tba
c = Counter()
for k in match_keys("v2"):
    r = load(k, "v2"); info = tba.match(k)
    for al in ("red", "blue"):
        tm = info.teams(al)
        segs = sorted(((e["slot"], e["t0"], e["t1"], e["team"], e["ocr"]) for e in r["identity"][al].values()))
        for a, b in zip(segs[:-1], segs[1:]):
            da, db = a[2] - a[1], b[2] - b[1]
            if a[0] != b[0] or not (0 <= b[1] - a[2] < 0.25) or not (5.0 <= da <= 10.2 and 5.0 <= db <= 10.2 and abs(da - db) < 0.4):
                continue
            ra, rb = sum(a[4]), sum(b[4])
            reads = "reads on new piece" if rb > 0.5 else ("reads only before" if ra > 0.5 else "no reads")
            if a[3] == b[3]:
                c[(reads, "keep")] += 1
            else:
                ok = rb > 0.5 and b[3] in tm and tm[int(np.argmax(b[4]))] == b[3]
                c[(reads, "flip, agrees with new piece's reads" if ok else "flip, unsupported")] += 1
tot = sum(c.values())
print(f"SEG_MAX piece boundaries: {tot}")
for reads in ("reads on new piece", "reads only before", "no reads"):
    n = sum(v for (r_, _), v in c.items() if r_ == reads)
    print(f"  {reads:20s} {n:6d}: " + ", ".join(f"{o} {100 * v / max(n, 1):.1f} %" for (r_, o), v in sorted(c.items()) if r_ == reads))
