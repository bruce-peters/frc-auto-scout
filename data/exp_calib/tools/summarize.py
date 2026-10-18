"""Tables for RESULTS_section.md from measure.json / measure_cpu.json (no Python state needed).

The 'realistic' set takes measure_cpu.json where present (new bumper reads made on the CPU, as a real
reanalyse would make them) and measure.json otherwise (those scans need no new reads, or have no crop
images to read). Prints markdown.
"""
import json
from pathlib import Path

import numpy as np

EXP = Path(r"C:\Users\bruce\Documents\Coding\auto-scout\data\exp_calib")
AFFECTED = {"2026cagle_qm32": "red", "2026cagle_sf13m1": "red", "2026cancmp_qm114": "blue",
            "2026cancmp_qm52": "blue", "2026casnv_qm7": "blue", "2026cc_qm15": "blue", "2026joh_qm14": "blue"}
none = json.loads((EXP / "measure.json").read_text())
cpu = json.loads((EXP / "measure_cpu.json").read_text()) if (EXP / "measure_cpu.json").exists() else {}


def table(data, label):
    print(f"\n### {label}\n")
    print("| match | alliance | recovered view | share err vs OPR | credited launches (own-end view) | held-out identity |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    agg = {k: [] for k in ("eb", "ea", "cb", "ca", "ob", "oa")}
    hid = {"b": [0, 0], "a": [0, 0], "ab": [0, 0], "aa": [0, 0]}
    for key, al in AFFECTED.items():
        r = data[key]
        b, a = r["before"]["alliances"][al], r["after"]["alliances"][al]
        rec = r["after"]["calibrated"].get(f"{al}_end", "-")
        hb, ha = r["before"]["holdout"][al], r["after"]["holdout"][al]
        print(f"| {key} | {al} | {rec if rec != 'tags' else '-'} | {b['share_err_opr']:.3f} -> {a['share_err_opr']:.3f} | "
              f"{b['credited']} ({b['credited_own_end']}) -> {a['credited']} ({a['credited_own_end']}) | "
              f"{100 * hb['agree'] / hb['mass']:.1f} -> {100 * ha['agree'] / ha['mass']:.1f} % |")
        agg["eb"].append(b["share_err_opr"]); agg["ea"].append(a["share_err_opr"])
        agg["cb"].append(b["credited"]); agg["ca"].append(a["credited"])
        other = "red" if al == "blue" else "blue"
        agg["ob"].append(r["before"]["alliances"][other]["share_err_opr"])
        agg["oa"].append(r["after"]["alliances"][other]["share_err_opr"])
        for s, st in (("ab", "before"), ("aa", "after")):
            h = r[st]["holdout"][al]
            hid[s][0] += h["agree"]; hid[s][1] += h["mass"]
        for s, st in (("b", "before"), ("a", "after")):
            for x in ("red", "blue"):
                h = r[st]["holdout"][x]
                hid[s][0] += h["agree"]; hid[s][1] += h["mass"]
    rec6 = [i for i, k in enumerate(AFFECTED) if k != "2026joh_qm14"]
    print(f"\n* affected alliances, mean share error vs fuel OPR: {np.mean(agg['eb']):.3f} -> {np.mean(agg['ea']):.3f} "
          f"(the 6 recovered: {np.mean([agg['eb'][i] for i in rec6]):.3f} -> {np.mean([agg['ea'][i] for i in rec6]):.3f}; "
          f"better in {sum(agg['ea'][i] < agg['eb'][i] - 1e-9 for i in rec6)}, worse in "
          f"{sum(agg['ea'][i] > agg['eb'][i] + 1e-9 for i in rec6)} of 6)")
    print(f"* the other alliance of the same matches: {np.mean(agg['ob']):.3f} -> {np.mean(agg['oa']):.3f}")
    print(f"* credited launches, affected alliances: {sum(agg['cb'])} -> {sum(agg['ca'])} "
          f"(median {np.median(agg['cb']):.0f} -> {np.median(agg['ca']):.0f})")
    print(f"* held-out identity, affected alliances pooled: {100 * hid['ab'][0] / hid['ab'][1]:.1f} -> "
          f"{100 * hid['aa'][0] / hid['aa'][1]:.1f} % (held-out read mass {hid['ab'][1]:.0f} -> {hid['aa'][1]:.0f}); "
          f"both alliances of the 7 matches: {100 * hid['b'][0] / hid['b'][1]:.1f} -> {100 * hid['a'][0] / hid['a'][1]:.1f} %")


table(none, "No new bumper reads (cached reads only, both states)")
if cpu:
    real = {k: cpu.get(k, none[k]) for k in AFFECTED}
    table(real, f"New bumper reads made on the CPU where the scan has crop images ({', '.join(sorted(cpu))})")
    for k in sorted(cpu):
        print(f"* {k}: {cpu[k]['after']['new_ocr_reads']} new reads in the after state, "
              f"{cpu[k]['before']['new_ocr_reads']} before")
