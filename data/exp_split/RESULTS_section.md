## Dense volleys: splitting chained flight tracks (2026-10-18, adopted with a refit credit model)

**Problem.** Bruce saw that 254's volleys barely register in the main camera. In 2026cc_qm24 at 70-74 s (40
scouted balls), the main view shows 254's balls clearly arcing to the blue hub, yet none of the ~117
main-view tracks near 254 was a shot. 50 were "slow", 34 "bad fit" and 33 "not rising". In a dense stream
the greedy flight tracker hooks each ball onto the one before it. A track becomes a 2-4 s zigzag over many
balls (up to 116 points), so its starting velocity is an average over several balls, and it fails the
projectile fit.

**Tried** (switches in `autoscout/pipeline.py`, both off):

* `SPLIT_TRACKS`: cut stored tracks where they hop to another ball (`fuel.split_at_jumps`) before the shot
  tests. In the qm24 window this gives 51 shots in the main view instead of 0. The launch gate then drops 45
  of them, because the first sighting of a split piece is 40-150 px above 254's 60 px box.
* `GATE_TRACE`: gate on where the arc, traced back along its initial velocity (≤ 0.3 s), crossed the box
  top, not on its first sighting.

**Result** (`data/exp_audit/tools/link_x_ab.py --arms`, 54 scouted Chezy quals, no-hindsight defaults, credit
model not refit):

| arm | video-only fuel MAE | video-only corr | top scorer | balls per jump given to robots scouts saw shooting |
| --- | ---: | ---: | ---: | ---: |
| base | 48.0 | 0.830 | 63 % | 63.5 % |
| split | 51.0 | 0.813 | 60 % | 66.0 % |
| split + traced gate | 51.1 | 0.812 | 61 % | 66.2 % |

* **Paired by match:** video-only MAE +3.0 (95 % CI −1.2 .. +7.3) and +3.1 (−1.4 .. +7.7). 32 and 31 of 54
  matches get worse.
* **Why:** splitting finds the missing launches, but in a crowded end of the field many are tied to the
  wrong robot. In qm24's main view 3847 stands in front of 254, and 5507, which scouts say did not shoot,
  goes from 8 to 45 balls. Per-jump credit improves slightly (burst column), but match totals get worse. This
  is the same pattern as the ballistic tracker (`data/exp_stream`): recall is not the bottleneck on its own.
* **Not tested above:** whether the refit credit model would change the picture (see below).

### Adopted: the same tracks with the credit model refit on them

The A/B above used the credit model fitted on the old, sparse launches. With 3-4x the launches, its balance
between launch evidence and its fallbacks (prior / hidden credit, the OPR pull) no longer fits. Refit on each
arm's own evidence (`data/exp_attrmodel/tools/build_ds.py data/exp_split/<arm> ... --season`, then `fit_cv.py`
at λ 3e-5, nested leave-one-match-out over 105 scouted alliances in 54 matches):

| arm (credit model refit on it) | share error (TV) | share corr | top scorer | fuel MAE |
| --- | ---: | ---: | ---: | ---: |
| today's tracks | 0.126 | 0.848 | 75 % | 41.9 |
| split | 0.118 | 0.872 | 75 % | 40.1 |
| **split + traced gate (adopted)** | **0.114** | **0.879** | **76 %** | **38.8** |

* **Paired by match, split + traced gate vs today:** fuel MAE −3.05 (95 % CI −5.95 .. −0.04) and TV −0.012
  (−0.020 .. −0.003). 31 matches are better and 23 worse.
* **The refit leans on the launches:** the evidence exponent b goes 0.6 → 0.97, so balls are credited almost in
  proportion to the arcs seen, and the OPR pull eases (0.55 → 0.46).
* **Ball-level recall** (tracker session, sole-shooter scoreboard episodes, first 18 matches): in 20+ ball
  volleys, about 0.19 of scored balls become correctly linked arcs with today's tracks, and about 0.70 split.
  Arcs credited to idle teammates rise in the same proportion; the refit model copes with that.
* **Defaults:** `SPLIT_TRACKS = GATE_TRACE = True` with `models/attr_model.json` refit on these tracks. The
  previous model is kept as `models/attr_model_v8.json`.
* **Later the same day:** the OPR prior was removed from the credit model by design (per-match evidence only).
  The evidence-only model on these tracks (`lin/all+dist`, `attr_model_evidence.json`) scores fuel MAE 42.5, top
  scorer 65 %, against the older hand-tuned rule's 51.1 / 61 %.

### Same ball in two cameras (checked on these tracks)

A robot seen by the main camera and an end camera gets launch evidence from both views. Each counter jump
is shared by evidence, so that robot could get more than its share; the alliance totals come from the
scoreboard and cannot double-count. Refit on these tracks (nested CV, λ 3e-5, `fit_cv.py` sets `vmax` /
`vsoft`):

| launch evidence per robot and window | share error (TV) | top scorer | fuel MAE |
| --- | ---: | ---: | ---: |
| sum over cameras (adopted) | 0.114 | 76 % | 38.8 |
| best camera only (each ball once) | 0.114 | 75 % | 38.6 |
| best camera + fitted share of the others (fits 0.59) | 0.114 | 76 % | 38.8 |
| separate weight per camera | 0.115 | 76 % | 39.0 |

* **The overlap is real but costs nothing measurable:** main-camera launches fit at about half the weight of
  own-end ones, and the extra cameras add about 60 % rather than 100 %.
* **Why it cancels:** teammates tend to be seen by the same mix of cameras, so the shares barely move.
* **Kept:** the sum.

### Stacked robots: credit the first box top the arc crosses (tested, not adopted)

In the main camera, a robot standing in front of another sits just below it in the image (3847 below 254 in
cc_qm24). An arc traced back past the shooter's box top carries on down to the lower box. Rule
`LINK_FIRST_CROSS`: among the candidate boxes the traced arc crosses, credit the first one going back in time.

* **All cameras:** the low end cameras hand balls to a taller robot standing beside the shooter (5507 36 → 60
  balls in qm24).
* **Main camera only** (`LINK_FIRST_CROSS_VIEWS`): evidence-only refit fuel MAE 42.5 → 42.8, paired +0.24
  (CI −0.40 .. +0.91), 25 matches better and 29 worse. Left off.
