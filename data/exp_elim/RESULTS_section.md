## Process of elimination for unidentified robots and launches (2026-10-19, adopted)

**Idea (Bruce).** When the other robots of an alliance are seen somewhere else, an unidentified robot, or an
unclaimed launch, must belong to the remaining one.

**Why it was needed.** In 2026cc_qm17, 6800 is the blue alliance's top scorer. Scouts count about 350 shots;
it "dumps a full hopper in 3-4 s" from right under the blue end camera. It still got 0 linked launches, for
two reasons:
* **Mislabelled track.** The causal DINO probe had barely seen 6800, so it called 6800's track "6665" at full
  confidence. The identity Viterbi then labelled two tracks metres apart "6665" for 30 s and marked 6800
  absent.
* **No box at all.** 6800 sits too close to the camera to be boxed, so its arcs had no box and no tracked
  robot near them, and were dropped.

6800's 286 credited balls came entirely from the credit model's hidden-robot pseudo-shots. Meanwhile 6665,
pure defence and 0 shots, had 43 launches pinned on it.

**What changed** (`autoscout/pipeline.py`, `identity.assign_absent`):

| piece | switch | rule |
| --- | --- | --- |
| tracks | `IDENT_FAR_COST = 20` (`IDENT_FAR_M = 2.5`) | Two concurrent tracks of one alliance more than 2.5 m apart pay 20 per epoch to share a label. The mirror image of the co-located duplicate bonus. |
| "?" boxes | `ELIM_BOXES` | A box with no field position (image edge, far down an end camera) gets the one team of its alliance not labelled on another box in the frame and not tracked elsewhere (projects more than `ELIM_FAR` = 1.5 box widths away). |
| box-less arcs | `ELIM_ARCS` | A clean rising arc with no box or tracked robot near it, launched inside an alliance zone, goes to that alliance's one robot not tracked more than 2 m from the launch point. Confidence 0.6, `how: "elim"`. |
| with them | `BOX_HOLD_S = 1.5`, `LINK_FILL_TIE = 3` | From the visual QA (`data/exp_visual_qa/report.md`). A segment's last box stays a link candidate for 1.5 s after the detector drops it. Between overlapping boxes, the robot whose hopper fill drops across the launch wins. |

* **qm17 with everything on:** 6800 gets 181 launches, 435 box-less arcs are claimed, and 53 "?" boxes are
  named.
* **Inspector:** every inferred team, counted or not, is drawn dashed with a hollow number plate, under the
  layer "Inferred teams".

**Scouting A/B.** The 65 scouted Chezy quals were analysed in memory, nine arms
(`data/exp_audit/tools/link_x_ab.py`, `score_arms.py`). The credit model was refit on each arm's evidence:
evidence only, lin/all+dist at λ 3e-5, nested leave-one-match-out over 105 alliances in 54 matches. Paired
bootstrap over matches (`data/exp_attrmodel/tools/paired_cv.py`).

| arm (refit, held out) | share error (TV) | top scorer | fuel MAE | vs today (95 % CI) | hidden pseudo-shots |
| --- | ---: | ---: | ---: | --- | ---: |
| today | 0.121 | 64 % | 42.4 | | 9.7 |
| box-less arcs only | 0.120 | 66 % | 41.9 | −0.51 (−2.94 .. +1.16) | 6.0 |
| all three elimination pieces | 0.120 | 67 % | 42.0 | −0.38 (−2.72 .. +1.55) | 6.2 |
| **elimination + box hold + fill tie (adopted)** | **0.117** | **69 %** | **41.2** | **−1.20 (−3.59 .. +0.67)** | **6.2** |
| box hold alone | 0.120 | 65 % | 42.4 | −0.02 (−0.86 .. +0.83) | 9.7 |
| fill tie alone | 0.120 | 64 % | 42.3 | −0.14 (−0.47 .. +0.16) | 9.6 |
| hidden robots' projection radius 2.5 m | 0.121 | 64 % | 42.3 | −0.07 (−0.39 .. +0.26) | 9.0 |

* **Share error** with the adopted arm: −0.0046 (95 % CI −0.0104 .. +0.0003).
* **Fuel to robots with no linked launch** all match: 0.5 % today, 0 with elimination.
* **The hidden-robot term shrinks by a third.** With the shooter's arcs found, the model needs to guess less
  about unseen robots.
* **Why adopted although within noise.** The direction is the same across arms, and all three measures
  improve together. It also gives credit from observed launches instead of a guess.

**Launch-only credit (not adopted).** Bruce asked why a robot with no launches gets any balls. The test:
`attribution.CREDIT_RULE = "launches"` gives every scoreboard jump only to robots seen launching, nearest
launches first, with no hidden-robot pseudo-shots and no fitted weights. Scored directly with nothing fitted:

| arm | fuel MAE | top scorer | vs today (95 % CI) |
| --- | ---: | ---: | --- |
| launch-only | 57.7 | 62 % | +16.1 (+8.7 .. +24.0) |
| launch-only + elimination | 50.3 | 66 % | +8.7 (+3.2 .. +14.5) |

Some shooters are still never seen launching, so the hidden-robot term is still worth about 8 balls per
robot-match. It stays, a third smaller. `CREDIT_RULE` is kept as a switch.

**Model.** `models/attr_model.json` = `data/exp_attrmodel/attr_model_elim.json`: prior 0.031, hidden 6.16,
b 1.32, zone power −0.07, g_dist −0.77, g_opr 0. Previous model: `attr_model_evidence_zone.json`.
