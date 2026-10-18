## Stream separation: ballistic flight tracker (tried 2026-10-18, not adopted)

Volleys are the largest per-video error class: credited launch arcs per scouted ball fall about 5x from small
bursts to 20+ ball volleys (failure-finder cycle 17). In a stream the greedy flight tracker loses most arcs: a
new one-point track predicts zero velocity and grabs the *following* ball, and its wide fixed gate lets tracks
hop between balls. `fuel.BallisticTracker` (opt-in: `Options.flight_tracker = "ballistic"`) uses speed-scaled
gates, one optimal assignment per frame and a two-frame look-ahead to seed new tracks. On a synthetic 20-30
balls/s stream it gets 12/12 smooth arcs vs 0/12 for the greedy tracker.

Evaluated with `scripts/stream_experiment.py` / `scripts/stream_resplit.py` on the v3 scans, re-tracked
offline from per-frame blob logs, with identical analysis code (2026-10-18 08:00) for every arm. There are
three match sets:

* 82 matches: all 65 scouted 2026cc quals, 4 cc playoffs and 13 2026joh matches.
* 81 matches for the burst counts: the same set without 2026cc_qm14, whose blob log used a wrong match window.
* 103 fully scouted alliances in 53 matches for the scouting comparisons.

The flying-fuel detector variants come from the A/B above.

**Launch recall per scouted ball, by which camera sees the shooter** (82 matches). A ball counts as detected
if a credited hub-bound launch of that team lies within ±1 s; the failure-finder cycle 11 definition.

| arm | own end view | main view only | no camera | all |
| --- | ---: | ---: | ---: | ---: |
| greedy (default) | 0.63 | 0.28 | 0.15 | 0.42 |
| ballistic | 0.81 | 0.55 | 0.35 | 0.63 |
| motion + flying-fuel detector, ballistic | 0.82 | 0.57 | 0.37 | 0.65 |
| flying-fuel detector only, ballistic | 0.76 | 0.38 | 0.23 | 0.53 |

**Credited launches per scouted ball, by scouted burst size** (81 matches; launches seen by two cameras within
0.1 s are counted once):

| burst | balls | greedy | ballistic |
| --- | ---: | ---: | ---: |
| 1-4 | 851 | 0.38 | 1.27 |
| 5-9 | 3,557 | 0.17 | 0.58 |
| 10-19 | 9,965 | 0.14 | 0.47 |
| 20+ | 26,645 | 0.08 | 0.29 |
| 20+ relative to 1-4 | | 0.21 | 0.22 |

* 5026 in 20+ bursts: 0.12 (greedy) -> 0.34 (ballistic).

The ballistic tracker multiplies launches 3.3-4.2x at every burst size, so volleys stay about 5x
under-credited relative to small bursts.

**Team level.** Scout TV = mean 0.5 · Σ|video share − scouted share| per alliance; the 5026 gap is the video
share minus the scouted share of 5026. Columns are TV / share corr / top scorer / 5026 gap.

| credit rule | greedy | ballistic |
| --- | --- | --- |
| hand-tuned rule (pure video) | 0.153 / 0.777 / 61 % / −0.079 | 0.158 / 0.769 / 54 % / −0.080 |
| learned model as deployed (fitted on greedy evidence) | 0.133 / 0.836 / 67 % / −0.069 | 0.134 / 0.837 / 62 % / −0.074 |
| learned model refitted on that arm (nested leave-one-match-out CV) | 0.133 / 0.830 / 64 % / −0.065 | 0.132 / 0.840 / 65 % / −0.068 |

* Fuel-OPR share corr, pure video, 82 matches: 0.519 (greedy) vs 0.527 (ballistic).
* Refitted fuel MAE: 44.1 (greedy) vs 44.3 (ballistic).
* Ballistic minus greedy, both refitted, paired bootstrap over matches: ΔTV −0.001 (90 % CI −0.008 to
  +0.005), Δcorr +0.009 (−0.006 to +0.025).
* Merging a team's launches from different cameras within 0.1 s changes TV by 0.002 or less.

Conclusion: the ballistic tracker fixes launch *recall*, nearly doubling it for shooters seen only by the main
camera, but not team attribution. Its extra launches spread over every robot and burst size in proportion, so
shares do not move beyond what the learned credit model already gives, even with the model refitted on its
evidence. Volleys are still under-credited relative to small bursts. The detector does not help in the main
view, where the balls are too small for it. The greedy tracker stays the default; adopting the ballistic
tracker would need a fuel-only re-scan of every match (`pipeline.rescan_fuel`, CPU only).
