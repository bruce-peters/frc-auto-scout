"""Is the low recall of fast shooters an under-credit? video fuel / scouted shots and share residuals
by scouted shooting-rate tercile (cc). Plus burst listing for one robot."""
import json, sys
import numpy as np
from common import AUDIT, load, tba, TL
rows = json.load(open(AUDIT / (sys.argv[1] if len(sys.argv) > 1 else "c3c_recall_cc_v2.json")))
sc = json.load(open(AUDIT / "scout_ref_cc_v2.json"))["cc_v2"]["rows"]
idx = {(x["match"], x["team"]): x for x in sc}
by_al = {}
for x in sc:
    by_al.setdefault((x["match"], x["alliance"]), []).append(x)
rows = [x for x in rows if (x["match"], x["team"]) in idx]
for x in rows:
    s = idx[(x["match"], x["team"])]
    al = by_al[(x["match"], s["alliance"])]
    vs = np.array([y["video"] for y in al]); ss = np.array([y["scout"] for y in al]); os_ = np.array([y["opr"] for y in al])
    i = [y["team"] for y in al].index(x["team"])
    x["ratio"] = s["video"] / max(s["scout"], 1)
    x["res_scout"] = vs[i] / vs.sum() - ss[i] / ss.sum()
    x["res_opr"] = vs[i] / vs.sum() - os_[i] / os_.sum()
    x["scout_share"] = ss[i] / ss.sum()
order = np.argsort([x["rate_active"] for x in rows])
n = len(order)
for name, ids in (("slowest", order[:n // 3]), ("middle", order[n // 3:2 * n // 3]), ("fastest", order[2 * n // 3:])):
    g = [rows[i] for i in ids]
    print(f"{name:8s} rate {np.median([x['rate_active'] for x in g]):4.1f}/s  video/scout {np.median([x['ratio'] for x in g]):.2f}  "
          f"share resid vs scout {np.mean([x['res_scout'] for x in g]):+.3f}  vs OPR {np.mean([x['res_opr'] for x in g]):+.3f}  "
          f"scout share {np.mean([x['scout_share'] for x in g]):.2f}")
