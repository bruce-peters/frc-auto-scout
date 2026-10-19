## Season-to-date fuel prior for the credit model

**Question.** With no hindsight, the learned credit model's fuel-OPR prior uses only the quals already played
at the event. The first match has no prior at all, and the next ones have a noisy one. Every team's earlier
events of the season are public before an event starts. Can those events fill the gap, alone or blended with
the in-event OPR, and does that lower the credit model's error against 5026's Chezy scouting?

**Answer.** Yes for the blend, which is recommended for adoption. With the blend, fuel MAE drops from 43.6 to
41.4 balls (nested CV). The 95 % interval of the difference, −4.2 … −0.3, excludes zero. Share error (TV) goes
from 0.129 to 0.124, with an interval of −0.010 … +0.001.

* **Where the gain comes from.** Almost all of it is in quals 21+ (43.8 → 40.8). On quals 1–20 the blend is
  42.5 against 43.2, too few matches to tell apart. The in-order prior's early gap to the hindsight
  full-event OPR is +1.0 balls here (43.2 vs 42.2); the blend closes about 0.7 of it.
* **Beyond hindsight.** The blend also beats the full-event OPR (42.4), which a live system cannot have.
  Ten quals per team is not much data, and the earlier events add about 7 more quals' worth per team.
* **The earlier events alone.** The season rate without any in-event data (42.6) is between the two and not
  distinguishable from the in-order prior.
* **Coverage.** All 42 Chezy teams have earlier 2026 events with TBA hub totals: 3–7 events each, a median of
  7.2 recency-weighted quals. With the in-order prior, 14 of the 105 scouted alliances have no prior, because
  a team had not played yet. With the blend, none do.

### Method (`autoscout/season.py`)

* **`season_rate(team, event_key)`.** The team's fuel OPR at each earlier event of the season, computed with
  `opr.fuel_opr` over that event's full quals from TBA hub totals.
  * Only events that ended before `event_key`'s start date count. Preseason events (type 100) and events
    with fewer than 8 scored quals are skipped.
  * Each event is weighted by quals played × 0.5^(days before the event / 60).
* **`prior_oprs(event_key, before=match_key)`.** This is the drop-in for `opr.fuel_opr(event, before=)`. It
  is a ridge OPR over the event's quals played before the match, shrunk toward each team's season rate.
  * The shrink strength is 0.5 alliance-match observations per earlier qual (`SEASON_WEIGHT`).
  * One least-squares factor, fitted to the alliance totals played so far, puts the season rates on the
    event's scale.
  * Before the first match, it returns the season rates themselves.
  * A team with no season rate gets the event mean, with `opr.fuel_opr`'s own ridge (0.5).
* **`season_prior(team, event_key, before_match)`.** One team's entry of `prior_oprs`.
* **Statbotics.** The v3 API returned HTTP 500 (once 503) for 2026 `team_year`, `team_event` and `year`,
  and for 2025, on both tries during this work. `statbotics_rate` is wired in as a fallback for teams with
  no TBA history. It is cached in `data/cache/statbotics` and returns None on any failure. Its 2026 field
  names are unverified, and Chezy never needed it.
* **Caching.** TBA responses are cached by `tba.get` in `data/cache/tba`. A cold start at Chezy needs 42
  `team_events` calls plus 69 event match lists, about 2 minutes at the rate TBA served today.

### Tuning (`tune.py`; Champs divisions and summer offseasons, no Chezy data)

The half-life, the normalisation and the season weight were picked on two other sets before Chezy was looked
at:
* the 8 Champs divisions: 597 teams, predicted from pre-Champs events;
* 14 summer offseasons with ≥ 20 teams and ≥ 40 scored quals, 1–4 months after the season, like Chezy: 463
  teams, 443 of them with history.

**T1, event start.** Correlation of the season rate with the event's full-quals fuel OPR, averaged over
events.

| half-life (days) | 14 | 30 | 60 | 120 | ∞ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Champs divisions, raw OPR | 0.780 | 0.778 | 0.775 | 0.773 | 0.771 |
| Champs divisions, level-normalised | 0.753 | 0.753 | 0.751 | 0.749 | 0.746 |
| summer offseasons, raw OPR | 0.723 | 0.732 | **0.733** | 0.732 | 0.729 |
| summer offseasons, level-normalised | 0.702 | 0.709 | 0.710 | 0.709 | 0.707 |

* **Normalisation.** "Level-normalised" divides each earlier event's OPR by the event's effect in a fit of
  log OPR = team effect + event effect (`event_levels`, `NORM`). It is worse on both sets, so the raw OPR
  is used.
* **Half-life.** It barely matters. 60 days is the offseason optimum.
* **Chezy check (not used for tuning).** The season rate correlates 0.52 with Chezy's final fuel OPR, and
  0.65 with each team's mean scouted shots. The full-event OPR correlates 0.88 with the scouted shots, but it
  comes from the same matches.

**T2, in order.** Before every qual: the red/blue fuel ratio predicted from the prior available then,
against the actual ratio. The metric is the RMS of the log ratio error. A team not yet seen gets the mean.
Divisions use a 14-day half-life, offseasons 60 days, raw OPR.

| prior | divisions q1–20 | divisions q21+ | offseasons q1–20 | offseasons q21+ |
| --- | ---: | ---: | ---: | ---: |
| in-event OPR only (pipeline today) | 0.587 | 0.461 | 1.123 | 0.830 |
| season rate only | 0.444 | 0.421 | 0.911 | 0.840 |
| blend, season weight 0.25 | 0.444 | 0.407 | 0.943 | 0.767 |
| **blend, season weight 0.5 (adopted)** | 0.442 | 0.397 | 0.922 | **0.761** |
| blend, season weight 1 | 0.441 | **0.393** | 0.909 | 0.769 |
| blend, season weight 2 | 0.441 | 0.394 | **0.907** | 0.783 |

* **Divisions.** The earlier events beat the in-event OPR even late at Champs, and the blend beats both.
* **Offseasons.** Late in an event the in-event matches matter more, so the optimum weight is lower (0.5 to
  1). 0.5 was taken because Chezy's gap to the season is even longer.

### Results on Chezy (`eval_cc.py`, `eval_cc.log`, `eval_cc.json`)

**Setup.**
* `lin/all+dist+mlprior` is refitted leave-one-match-out on `ds_v6.pkl` (105 scouted alliances, 54 matches),
  with each alliance's OPR swapped for the prior under test.
* The stored in-order OPR equals `opr.fuel_opr(before=)` exactly, and that row reproduces `cv_v6.log`.
* **Nested** means λ is picked inside each training fold. **Matched** means λ = 3e-5 for every prior, the
  deployed value.
* Quals 1–20 are 38 alliances in 19 matches.

| prior | CV | TV all | TV q1–20 | TV q21+ | fuel MAE all | MAE q1–20 | MAE q21+ | corr | top | g_opr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| in-order OPR (pipeline today) | nested | 0.129 | 0.131 | 0.127 | 43.6 | 43.2 | 43.8 | 0.844 | 72 % | 0.27 |
| season rate only | nested | 0.127 | 0.132 | 0.124 | 42.6 | 43.4 | 42.1 | 0.848 | 69 % | 0.52 |
| **blend (adopted)** | nested | **0.124** | 0.131 | **0.121** | **41.4** | 42.5 | **40.8** | **0.855** | **77 %** | 0.50 |
| full-event OPR (hindsight, not allowed live) | nested | 0.125 | 0.130 | 0.122 | 42.4 | 42.2 | 42.6 | 0.851 | 71 % | 0.28 |
| in-order OPR | matched | 0.128 | 0.131 | 0.126 | 43.2 | 43.2 | 43.2 | 0.847 | 71 % | |
| season rate only | matched | 0.126 | 0.130 | 0.123 | 42.2 | 43.0 | 41.8 | 0.850 | 70 % | |
| **blend (adopted)** | matched | **0.124** | 0.130 | **0.120** | **41.3** | 42.4 | **40.7** | **0.856** | **77 %** | |
| full-event OPR (hindsight) | matched | 0.125 | 0.130 | 0.122 | 42.4 | 42.0 | 42.6 | 0.852 | 71 % | |

**Differences from the in-order OPR** (95 % paired bootstrap intervals over matches):

| prior | CV | subset | TV | fuel MAE | corr | top (points) |
| --- | --- | --- | --- | --- | --- | --- |
| blend | nested | all | −0.010 … +0.001 | **−4.2 … −0.3** | −0.001 … +0.025 | −1 … +11 |
| blend | nested | q1–20 | −0.012 … +0.008 | −4.9 … +2.7 | −0.022 … +0.040 | −11 … +16 |
| blend | matched | all | −0.009 … +0.001 | **−3.8 … −0.1** | −0.003 … +0.022 | 0 … +12 |
| blend | matched | q1–20 | −0.012 … +0.008 | −5.2 … +2.5 | −0.022 … +0.042 | −11 … +16 |
| season only | nested | all | −0.010 … +0.005 | −3.6 … +1.3 | −0.014 … +0.022 | −11 … +2 |
| season only | nested | q1–20 | −0.011 … +0.009 | −4.2 … +3.5 | −0.032 … +0.035 | −13 … +13 |
| full-event OPR | nested | all | −0.011 … +0.003 | −3.1 … +0.9 | −0.013 … +0.028 | −7 … +5 |

* **Season weight sensitivity** (matched λ; the weight was picked on the tuning sets):

  | season weight | fuel MAE all | MAE q1–20 |
  | ---: | ---: | ---: |
  | 0.25 | 41.9 | 42.0 |
  | 0.5 | 41.3 | 42.4 |
  | 1 | 41.4 | 42.8 |
  | 2 | 41.5 | 43.0 |

  The result does not hinge on the exact weight.
* **The refit trusts the blended prior more.** The prior power `g_opr` goes from 0.27 to 0.50. The gain
  therefore comes from a better prior and the model leaning on it harder, together. A plain 0.7 video + 0.3
  prior split (no refit) also improves with the blend: fuel MAE 46.0 → 45.2.

### Recommendation: adopt the blend

**Port to `pipeline._attr_opr`** (the main session owns the file):

    from . import season
    _EVENT_OPR[k] = (season.prior_oprs(info.event_key, before=info.key) if NO_HINDSIGHT
                     else oprmod.fuel_opr(info.event_key))

**Refit the model.** Switch `models/attr_model.json` to `data/exp_season/attr_model_season.json`:
* fitted at λ 3e-5 on `ds_v6_season.pkl`, which is `ds_v6.pkl` with every alliance's OPR replaced by
  `prior_oprs` (`make_ds.py`, all 260 alliances);
* `g_opr` 0.50, `b` 0.59, `hidden` 1.18, `prior` 0.021, `gz` 0.42, `g_dist` −0.44;
* in-sample TV 0.120, fuel MAE 40.3.

Porting the prior without this refit keeps `g_opr` at 0.29, a combination that was not tested.

**Tests.** `tests/test_season.py` covers the earlier-event filter, the event-start values, and the limits of
the blend: at weight → 0 it is the event's own OPR, at weight → ∞ the rescaled season rates. `pytest tests -q`:
43 passed.

### Caveats

* **One event, one scouting team.** The early-qual subset is 19 matches, so its intervals are wide. The
  overall gain clears zero only on fuel MAE, by a small margin.
* **Chezy is a hard case for the season prior.** It is 4.5 months after the last official events, and
  offseason teams change drivers and robots. The season rate correlates 0.52 with Chezy's final fuel OPR,
  against 0.78 at Champs. The blend still helps there, and the tuning sets suggest it helps more in season.
* **Early-season events get little from it.** At week 2, casnv has 3 of 37 teams with earlier events, so the
  prior falls back to the in-event OPR. Coverage is 33/37 at cagle (week 4), 61/61 at cancmp and 74/75 at joh.
* **Transfer not re-run.** `transfer.py`'s split-half design (half-event OPR as the prior, the other half as
  the reference) does not fit an in-order prior. The new model was not re-checked on the other four events.
* **Dataset.** The brief mentioned a `v7_snapshot`. There is none, so this uses `ds_v6.pkl` (`v6_snapshot`,
  in-order OPR, matching `cv_v6.log`). Re-run `eval_cc.py` and `make_ds.py` on the next dataset before
  exporting.
* **Statbotics fallback untested.** The API was down throughout.

### A/B: Statbotics-style fuel EPA vs TBA fuel OPR as the season rate (`season.SOURCE`)

**Question.** Statbotics' 2026 EPA breaks out per-team fuel (`total_fuel`), and EPA predicts winners better
than OPR in their benchmarks. Is EPA as of the event start a better season rate than the deployed
recency-weighted fuel OPR, alone and blended with the event's matches?

**Statbotics itself was down.**
* The v3 API answered `/v3` with 200, but every data endpoint returned 500 or 503: `team_event`,
  `team_events`, `team_year`, `team_matches`, `event`, `year`, and 2024/2025 too.
* statbotics.io rendered a blank page.
* A 10-minute poll from 16:56 kept getting 500/503.
* The A/B therefore uses **Statbotics' EPA update re-run on TBA hub totals** (`season.season_epa`, source
  `"epa"`). It copies `backend/src/models/epa/main.py`:
  * the alliance's fuel error is split evenly over its 3 teams;
  * each EPA moves by 1/3 of its share for a team's first 6 quals, falling to 0.2 by its 12th;
  * playoffs count 1/3 and do not advance the qual count;
  * the margin term is 0 for 2026.
* It runs over all 22,913 scored 2026 matches (318 events, 3,779 teams) in match-time order. The rate is the
  team's EPA after its last match at an event that ended before the event started.
* Two differences from Statbotics. There is no carry-over from 2025. Every team starts at a third of the mean
  alliance fuel of the season's first 200 matches; Statbotics uses the whole season's mean, which is not
  causal.
* `source="statbotics"` reads the real API's end-of-event `total_fuel` (one `team_events` call per team).
  `eval_ab.py` adds it automatically once the API answers for all 42 Chezy teams.

**T1, event start.** Correlation with the event's full-quals fuel OPR, scored on the teams both sources cover.

| set | TBA fuel OPR | fuel EPA |
| --- | ---: | ---: |
| Champs divisions (595 teams) | 0.775 | **0.798** |
| summer offseasons (443 teams) | 0.733 | 0.736 |
| Chezy, vs final fuel OPR (42 teams; not used for tuning) | 0.511 | **0.598** |
| Chezy, vs mean scouted shots per team | 0.653 | **0.722** |

Coverage: EPA has a rate for 451 of 463 offseason teams, OPR for 443. Both cover 595 of 597 division teams and
all 42 Chezy teams.

**T2, in order.** RMS of the log red/blue fuel-ratio error before each qual. Half-life 60 days.

| prior | divisions q1–20 | divisions q21+ | offseasons q1–20 | offseasons q21+ |
| --- | ---: | ---: | ---: | ---: |
| in-event OPR only | 0.587 | 0.461 | 1.123 | 0.830 |
| OPR season rate only | 0.444 | 0.422 | **0.911** | 0.840 |
| EPA season rate only | 0.425 | 0.419 | 0.919 | 0.845 |
| OPR blend, weight 0.5 (deployed) | 0.441 | **0.393** | 0.922 | **0.761** |
| EPA blend, weight 0.5 | **0.422** | **0.393** | 0.921 | 0.771 |
| OPR blend, weight 1 | 0.442 | 0.397 | 0.909 | 0.769 |
| EPA blend, weight 1 | 0.423 | 0.397 | 0.916 | 0.775 |

**Chezy credit model** (`eval_ab.py`, `eval_ab.log`, `eval_ab.json`).
* The model is `lin/all+dist+mlprior`, refitted leave-one-match-out.
* The dataset is `ds_v7s.pkl`: `build_ds.py v7_snapshot ... --season`, built 12:08. No file in `v7_snapshot`
  is newer, and its stored priors equal `season.prior_oprs` now (max difference 0), so it was reused rather
  than rebuilt.
* 105 alliances in 54 matches; quals 1–20 are 38 alliances.

| prior | CV | TV all | TV q1–20 | fuel MAE all | MAE q1–20 | corr | top | g_opr |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| in-order OPR | nested | 0.132 | 0.143 | 44.4 | 47.4 | 0.834 | 70 % | 0.29 |
| OPR season rate only | nested | 0.130 | 0.141 | 43.8 | 46.9 | 0.839 | 68 % | 0.55 |
| **OPR blend (deployed)** | nested | 0.126 | 0.139 | 42.0 | 45.4 | 0.847 | 74 % | 0.53 |
| EPA season rate only | nested | 0.127 | 0.135 | 42.4 | 44.3 | 0.850 | 70 % | 0.71 |
| EPA blend | nested | **0.125** | **0.135** | **41.5** | **44.1** | **0.852** | 73 % | 0.59 |
| in-order OPR | λ 3e-5 | 0.131 | 0.142 | 44.2 | 47.1 | 0.837 | 70 % | |
| OPR season rate only | λ 3e-5 | 0.129 | 0.140 | 43.3 | 46.3 | 0.841 | 69 % | |
| **OPR blend (deployed)** | λ 3e-5 | 0.126 | 0.138 | 41.9 | 45.2 | 0.848 | 75 % | |
| EPA season rate only | λ 3e-5 | 0.126 | 0.134 | 42.2 | 43.9 | 0.850 | 70 % | |
| EPA blend | λ 3e-5 | **0.125** | 0.135 | **41.4** | **43.8** | **0.853** | 75 % | |

**Differences** (95 % paired bootstrap over matches):

| difference | CV | subset | fuel MAE | TV | corr |
| --- | --- | --- | --- | --- | --- |
| EPA only − OPR only | nested | all | **−2.9 … −0.1** | −0.007 … +0.000 | **+0.004 … +0.019** |
| EPA only − OPR only | nested | q1–20 | **−5.0 … −0.3** | −0.013 … +0.000 | **+0.003 … +0.035** |
| EPA only − OPR only | λ 3e-5 | all | −2.4 … +0.2 | −0.006 … +0.001 | **+0.002 … +0.017** |
| EPA only − OPR only | λ 3e-5 | q1–20 | −4.9 … +0.1 | −0.013 … +0.001 | **+0.001 … +0.036** |
| EPA blend − OPR blend | nested | all | −1.3 … +0.4 | −0.004 … +0.001 | 0.000 … +0.011 |
| EPA blend − OPR blend | nested | q1–20 | −3.3 … +0.5 | −0.008 … +0.002 | −0.004 … +0.022 |
| EPA blend − OPR blend | λ 3e-5 | all | −1.4 … +0.4 | −0.004 … +0.002 | −0.000 … +0.011 |
| EPA blend − OPR blend | λ 3e-5 | q1–20 | −3.4 … +0.5 | −0.009 … +0.002 | −0.005 … +0.023 |
| OPR blend − in-order OPR | nested | all | **−4.4 … −0.5** | −0.011 … −0.000 | 0.000 … +0.027 |

**Verdict: do not switch the default yet.**
* **At the event start, EPA is the better rating.** It wins T1 at Champs and at Chezy, ties at the
  offseasons, and as the only prior it beats the OPR season rate on Chezy (nested fuel MAE −2.9 … −0.1).
* **Blended with the event's own matches, which is what runs, the gain shrinks to noise.** Overall it is
  −0.5 balls (−1.3 … +0.4). On quals 1–20 it is −1.3 (−3.3 … +0.5). On T2 it ties after qual 20, wins
  early at Champs (0.422 vs 0.441) and ties at the offseasons.
* **Switching costs more than the code change.**
  * The credit model needs a refit: `g_opr` 0.59 vs 0.53.
  * The prior needs every event of the season: 318 match lists, 54 MB, about 5 minutes cold, against
    about 110 calls for the OPR rate at Chezy.
* **To switch later:** set `season.SOURCE = "epa"`, rebuild with `--season`, and re-export.
* **What would settle it:** a second scouted event, or the real Statbotics numbers once its API is back
  (`eval_ab.py` and `tune.py SET --ab opr,epa,statbotics`).

### Files (`data/exp_season/`)

* `fetch_divisions.py`: warms the TBA cache for the Champs divisions.
* `tune.py`: T1 and T2 tuning; `tune_divs.log` / `.json` and `tune_off.log` / `.json` hold the results.
  `tune.py SET --ab opr,epa` runs the source A/B: `ab_divs.log`, `ab_off.log`, `ab_*.json`.
* `eval_cc.py`: Chezy credit-model evaluation; results in `eval_cc.log` / `.json`.
* `eval_ab.py`: the Chezy season-rate source A/B; results in `eval_ab.log` / `.json`.
* `make_ds.py`: writes `ds_v6_season.pkl`.
* `attr_model_season.json`: the model to deploy with the blend.

### Removed from the default (2026-10-18, by design)

Bruce: per-team fuel has to be an empirical measurement of how a team did in that match, so no OPR,
season-to-date or other previous-match data may move balls between teams. The deployed credit model is now
`lin/all+dist`, fitted without the prior (g_opr 0). `season.py` and `pipeline.prior_opr` stay in the code, but
the default model doesn't call them.

Cost on the current split tracks (nested CV, λ 3e-5, 105 scouted alliances): fuel MAE 38.8 → 42.5, top scorer
76 → 65 %. The prior's gain came from nudging each team toward its usual level, which also flattens breakout and
off matches.
