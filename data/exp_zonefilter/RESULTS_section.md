## Launches from outside the robot's own zone are not shots (2026-10-18, adopted)

**Rule (Bruce).** In teleop a robot may only score from its own alliance zone (G407). A ball launched while
its robot is in the neutral zone or the opponents' half is a pass or a feed, so it should not count as launch
evidence at all.

**Before.** Zone was a property of the robot, not of the launch. Each scoreboard jump's weights multiplied a
robot's evidence by its zone fraction over the 4.7 s window. The refit credit model trusted that only weakly
(zone power 0.27), because positions near the zone line are noisy. Launches from outside the zone still counted.

**Now** (`pipeline.LAUNCH_ZONE_WEIGHT = 0`, `LAUNCH_ZONE_MARGIN = 1.0`): in teleop, a launch whose robot centre is
more than 1.0 m beyond its own zone line at the launch time is dropped. `result.json` counts them as
`shots_out_of_zone`.

* **How many:** 20.6 % of teleop launches at Chezy (split tracks) start more than 0.6 m beyond the robot's zone
  line, and almost all of those more than 1.5 m. The 1.0 m and 0.6 m margins give the same results.

**Scouting A/B** (54 scouted Chezy quals, credit model refit on each arm's evidence, evidence only with no OPR
prior, nested CV at λ 3e-5; `data/exp_attrmodel/tools/paired_cv.py`):

| arm | share error (TV) | top scorer | fuel MAE |
| --- | ---: | ---: | ---: |
| zone as a robot property (before) | 0.122 | 65 % | 42.5 |
| **launches from outside the zone dropped (adopted)** | **0.121** | 64 % | **42.4** |

* **Paired:** fuel MAE −0.15 (95 % CI −0.69 .. +0.43), TV −0.0005 (−0.0021 .. +0.0011). 35 matches are
  better and 19 worse.
* **Effect on the model:** the refit stops leaning on the robot-level zone factor (zone power 0.27 → 0.10).
  The per-launch rule does that job.
* **Why adopted although within noise:** it is the game rule, it costs nothing, and the inspector's launch
  arcs no longer include passes.
