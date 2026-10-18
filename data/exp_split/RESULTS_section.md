## Dense volleys in the main camera: splitting chained flight tracks (2026-10-18, not adopted)

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
* **Next:** these split arcs are the case the 3-D arc check (`data/exp_arcs`) must decide. The switches stay
  in the code so that it can be re-tested with that check.
