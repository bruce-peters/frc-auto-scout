## Launch-to-robot link between side-by-side robots (2026-10-18, adopted)

**Problem.** Bruce saw 254's volleys credited to a teammate in the inspector. In 2026cc_qm24 at 12.4 s, 254 is
shooting in the blue end camera, but most launch arcs went to 5507 next to it. A launch is tied to the robot
box it rises from. Each box is widened by 25 % on either side, so neighbouring boxes overlap. When an arc
starts inside two widened boxes, the tie was broken only by which box top it was nearest, never by left or
right. The taller neighbour won even when the ball rose out of the middle of 254's box.

**Fix.** The tie-break now also adds |u − box centre| / box width (`pipeline.LINK_X_TIE = 1`). Here u is
where the arc, traced back along its initial velocity, crossed that box's top edge (`LINK_X_CROSS`). Balls
drift sideways towards the hub as they rise, so the first point seen can sit over the neighbour.

**Test.** `data/exp_audit/tools/link_x_ab.py`. Each scan of the 54 scouted Chezy quals is analysed in memory
with every variant and scored against 5026's scouting with `scripts/score_scouting.py`. The credit model and
identity are the same in every arm (finalize_v5 defaults). The 0 arm reproduces `output/` exactly.

| tie-break | video-only fuel MAE | video-only corr | top scorer | best-estimate MAE |
| --- | ---: | ---: | ---: | ---: |
| none (before) | 48.8 | 0.825 | 61 % | 42.7 |
| first point seen, weight 1 | 46.8 | 0.838 | 65 % | 41.2 |
| **arc traced to the box top, weight 1 (adopted)** | **46.9** | **0.837** | **64 %** | **41.1** |
| arc traced to the box top, weight 2 | 46.9 | 0.837 | 64 % | 41.1 |

* **Paired by match:** video-only MAE −1.90 (95 % bootstrap CI −2.97 .. −0.93), best estimate −1.63
  (−2.55 .. −0.70). 35 matches are better and 18 worse.
* **What it does not fix:** 254 still loses 75 bursts (186 balls) to teammates, against 80 (189) before.
  Most of those bursts have few or no detected 254 launches, so the remaining problem is launch recall, not
  linking. Research into better per-robot launch counts is in `data/exp_launch/`.
