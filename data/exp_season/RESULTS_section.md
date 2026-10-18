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

### Files (`data/exp_season/`)

* `fetch_divisions.py`: warms the TBA cache for the Champs divisions.
* `tune.py`: T1 and T2 tuning; `tune_divs.log` / `.json` and `tune_off.log` / `.json` hold the results.
* `eval_cc.py`: Chezy credit-model evaluation; results in `eval_cc.log` / `.json`.
* `make_ds.py`: writes `ds_v6_season.pkl`.
* `attr_model_season.json`: the model to deploy with the blend.
