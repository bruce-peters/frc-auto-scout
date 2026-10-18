## Dead, absent and parked robots: absent identity state and parked-robot PRIOR (failure-finder cycle 16 follow-up)

**Question.** Robots the scouts saw doing nothing (dead, absent, parked, defence only) got 0.13-0.31 of their
alliance's fuel. Cycle 16 found two mechanisms: (1) `identity.assign` must use all three team labels, so a
never-read dead robot's label lands on a teammate's duplicate track and takes its launches (2026cc_qm18: the
box labelled 3045 shows "6647"); (2) a visible robot that sits still in its zone collects PRIOR on every
scoreboard jump (qm23 4499: 79 % of its credit was PRIOR).

### What was built (all off by default)

| piece | where | flag |
| --- | --- | --- |
| absent/unknown state: Viterbi over all 27 maps slot -> team. A team that labels no slot (active or inactive) is ABSENT for that epoch at a per-epoch cost; two slots with one label is a duplicate track. A team on an inactive slot is still just hidden (no cost). | `identity.assign_absent` | `Options.absent_cost_dead` (dead-flagged teams), `Options.absent_cost` (others); `None` = 6-permutation model |
| dead-robot evidence: < 0.5 own bumper evidence over the whole match while both teammates have >= 3, and its (OCR-only) labelled segments carry >= 1.0 of teammates' reads | `identity.dead_teams` | used when the absent state is on |
| co-located concurrent segments (< 1.5 m apart) favour one label | `assign_absent(near_m, near_bonus)` | `Options.absent_near_m`, `absent_near_bonus` |
| attribution: no PRIOR / HIDDEN inside a team's absent stretches (its launches still count) | `TeamSignals.absent`, `split_increments` | follows the absent state |
| dead-flagged team absent >= N s in total: no HIDDEN all match | `TeamSignals.no_hidden` | `Options.absent_dead_hidden_s` |
| parked robot: median over 20 s windows of the spread of its observed positions < `park_ext` m -> no PRIOR all match (HIDDEN and launches kept) | `pipeline._median_extent`, `TeamSignals.prior_off` | `Options.park_ext` (0 = off) |

`result.json` gains `absent`, `dead` and `parked` only when a flag is on. Tests:
`test_identity_absent_state_lets_duplicate_share_a_label`, `test_absent_or_parked_robot_gets_no_prior_or_hidden_credit`
(`pytest tests -q`: 40 passed). With the flags off the old code runs unchanged (`assign` is untouched; the new
branch in `split_increments` is taken only for a robot with `absent` / `prior_off` / `no_hidden` set). The
prototype (`data/exp_identity/tools/proto.py`, hooks around `assign` / `split_increments`) and the real code give
identical fuel for every variant on qm3, qm18 and qm23.

### How it was measured

`pipeline.analyse` in memory on every `output/<match>/scan.pkl` (131 matches; nothing written to `output/`),
all variants of a match on one scan load, so every comparison is paired
(`data/exp_identity/tools/absent_eval.py`, metrics `summarize.py`, robot table `cases.py`; runs in `runs/`).

* **Bumper reads missing from the scans.** `reanalyse` makes new reads on the GPU and never saves them into
  `scan.pkl`: 46.5k crops (~350 per match) are uncached, and EasyOCR on the CPU takes ~2.3 s per crop. Each
  segment's evidence missing from the cache is taken from the production `result.json` (`identity[..]["ocr"]`)
  and spread over that segment's uncached crops, so the normal analysis reproduces production identity
  (0 of 131 matches with a segmentation mismatch; base fuel within 0.5 of `result.json` for 127 of 131 matches).
  The held-out identity split of that injected evidence (by crop id parity) is approximate.
* **Scouting** (5026, Chezy quals): 104 alliances with all three robots scouted; 32 of them have a non-shooter
  (scouted share < 0.06, every scout <= 15 shots; cycle 16b's definition). Error = within-alliance share
  total variation (TV), mean over alliances; corr = pooled share correlation; MAE = fuel vs TBA total split by
  scouted shots (`score_scouting.py`'s reference).
* **Held-out identity**: `eval_analysis.py --holdout` logic (labels from half the reads, scored on the other half).
* **Fuel OPR**: pooled within-alliance share corr, 259 alliances (train = casnv + joh); at its noise floor.
* The main session changed `pipeline.py`, `overlay.py` and the scans during this work, so the base differs
  between runs; compare variants only within a run. Final run = code of 02:27, 18 Oct.

### Results (final run, 131 matches)

| variant | scout TV all (104) | corr | MAE | TV non-shooter (32) | TV other (72) | held-out identity | OPR corr all / train / test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| base | 0.153 | 0.776 | 52.1 | 0.173 | 0.144 | 91.33 % | 0.643 / 0.688 / 0.626 |
| absent state (`absent_cost_dead=2, absent_near_bonus=2`) | 0.153 | 0.778 | 52.4 | 0.171 | 0.145 | **91.68 %** | 0.643 / 0.688 / 0.625 |
| parked (`park_ext=1.5`) | 0.152 | 0.778 | **51.8** | 0.170 | 0.144 | = base | 0.644 / 0.687 / 0.627 |
| absent + parked | **0.152** | **0.781** | 52.0 | **0.168** | 0.145 | 91.68 % | 0.643 / 0.687 / 0.626 |
| absent + dead-hidden (`absent_dead_hidden_s=8`) | 0.154 | 0.773 | 52.9 | 0.170 | 0.147 | 91.68 % | 0.636 / 0.685 / 0.617 |
| absent + dead-hidden + parked | 0.153 | 0.776 | 52.5 | 0.167 | 0.147 | 91.68 % | 0.636 / 0.684 / 0.618 |

Paired TV change vs base (bootstrap 95 % CI over alliances; better / worse alliances):

| variant | all 104 | non-shooter 32 | other 72 |
| --- | --- | --- | --- |
| absent | +0.0000 [-0.0016, +0.0019] (5 / 4) | -0.0023 [-0.0058, 0.0000] (2 / 0) | +0.0011 [-0.0008, +0.0033] (3 / 4) |
| parked | **-0.0009 [-0.0021, -0.0001]** (5 / 1) | **-0.0031 [-0.0066, -0.0005]** (5 / 0) | +0.0001 (0 / 1) |
| absent + parked | -0.0009 [-0.0028, +0.0011] (10 / 5) | **-0.0054 [-0.0101, -0.0016]** (7 / 0) | +0.0011 [-0.0007, +0.0034] (3 / 5) |
| absent + dead-hidden | +0.0013 [-0.0020, +0.0064] (5 / 4) | -0.0031 (2 / 0) | +0.0032 [-0.0007, +0.0098] (3 / 4) |

* Held-out identity with the absent state: +0.31 pp per match (CI [+0.13, +0.56]), 19 matches better, 3 worse,
  109 unchanged. 21 of 262 alliance-matches have a dead-flagged team; 156 s absent in total.
* Sensitivity (later code, 65 cc matches): `absent_cost_dead` 4 instead of 2, or no co-location bonus, leaves
  only qm18 changed (non-shooter TV -0.0002); the co-location bonus is what moves qm3 6036. `park_ext` 1.0 and
  2.5 give the same scouting result as 1.5, because no scouted robot lies between 0.94 m and 3.9 m.
* Cycle 16's 8 "new" non-shooter alliances (TV 0.275 in the output of that time, 0.229 with the final run's
  base): absent 0.220, parked 0.223, absent + parked 0.214.

The problem robots (share of alliance fuel; scouted share first):

| robot | scouted | base | absent | parked | absent + parked | why |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| qm3 6036 (not on the field) | 0.00 | 0.37 | 0.33 | 0.37 | 0.33 | dead-flagged, 5 s absent; most credit is HIDDEN |
| qm18 3045 ("not working") | 0.00 | 0.19 | 0.17 | 0.19 | 0.17 | dead-flagged, 21 s absent (6647 duplicates relabelled) |
| qm19 4499 (died mid-field) | 0.00 | 0.01 | 0.01 | 0.00 | 0.00 | parked (0.45 m); today's scans already fixed it |
| qm23 4499 ("did not move") | 0.00 | 0.18 | 0.18 | **0.07** | 0.07 | parked (0.50 m) |
| qm30 5507 (still after auto) | 0.02 | 0.13 | 0.13 | **0.07** | 0.07 | parked (0.75 m) |
| qm24 5507 | 0.00 | 0.11 | 0.11 | 0.07 | 0.07 | parked (0.72 m) |
| qm39 3045 | 0.01 | 0.08 | 0.08 | 0.06 | 0.06 | parked (0.94 m) |
| qm17 6665 (full defence) | 0.00 | 0.17 | 0.17 | 0.17 | 0.17 | moving; credit is launches linked to it |
| qm1 2910 (never read, real) | 0.43 | 0.39 | **0.29** | 0.39 | 0.29 | dead-flagged by mistake |
| qm33 9496 (never read, real) | 0.39 | 0.37 | **0.32** | 0.37 | 0.32 | dead-flagged by mistake (0.11 with dead-hidden) |
| qm38 9128 (scouted 40 shots) | 0.16 | 0.12 | 0.12 | 0.11 | 0.11 | parked (0.59 m), mostly hidden: keeps HIDDEN |

### Findings

1. **The absent state fixes identity, not credit.** It relabels the duplicate tracks the reads name (held-out
   identity +0.3 pp, CI excludes 0), but the forced label was not the main source of a dead robot's fuel. A
   robot that is not there is usually *hidden* in the model (its label rides an inactive slot), and hidden
   robots earn HIDDEN credit; the identity model has no evidence that tells "hidden" from "not there"
   (6036: 75 % of its credit was HIDDEN, only 5 s became absent).
2. **The dead-robot flag cannot tell dead from unread.** In the scouted alliances it flags 2 dead robots
   (3045, 6036) and 6 robots that did shoot (scouted share 0.07-0.43) but whose bumpers were never read (2910,
   9496, 6017, 1540, 5507, 581). Their labelled segments carry teammates' reads just as often (probably
   misreads or a neighbour's bumper in the crop). Relabelling those segments costs 2910 0.10 and 9496 0.05 of
   share, which cancels the gain on 3045 and 6036.
   Zeroing HIDDEN for flagged teams (`absent_dead_hidden_s`) takes 3045 to 0.05 but 9496 to 0.11 (scouted
   0.39) and lowers OPR corr by 0.007: rejected. Per-team visible time and co-location with teammates
   separate 6036 from the rest, but with one example that would be a fitted rule.
3. **Parked robots are easy to see and safe to fix.** The median 20 s spread of a robot's observed positions is
   < 1.0 m for 13 of 786 team-matches, 1.0-1.5 m for 3 and 1.5-3.5 m for 8 (all unscouted, cagle / casnv /
   cancmp); the 5th percentile is 4.5 m. In the scouted cc quals the parked cluster (<= 0.94 m) is separated
   from everyone else (>= 3.9 m). All 5 scouted parked non-shooters lose credit; the one parked robot that did
   shoot (qm38 9128, 40 shots while standing still, mostly hidden) keeps its HIDDEN and launches and moves
   0.01. Zeroing HIDDEN as well cost it 0.08 (prototype), so `park_ext` removes PRIOR only. Launches linked to
   a parked robot stay, which is why qm23 4499 keeps 0.07.
4. Remaining over-credit of non-shooters is launch evidence linked to the wrong robot (qm17 6665 defence,
   qm12), not identity or PRIOR.

### Recommendation

* **Adopt the parked-robot flag, `Options.park_ext = 1.0`.** At 1.5 (measured on all 131 matches) scouting
  improves on every non-shooter it touches (5 better, 0 worse), share TV -0.0009 [-0.0021, -0.0001] on all 104
  alliances, OPR unchanged (0.643 -> 0.644). 1.0 gives the same scouting result and leaves out the three
  unscouted robots at 1.0-1.3 m that nothing validates.
* **Absent state: optional, not for fuel.** `absent_cost_dead = 2.0, absent_near_bonus = 2.0` raises held-out
  identity by 0.3 pp and is neutral on fuel share (scouting and OPR); turn it on if per-team paths/identity
  matter. It needs a better dead-vs-unread signal before it helps attribution.
* Keep `absent_dead_hidden_s` and `absent_cost` (non-dead teams) off.
