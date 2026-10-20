## Per-team fuel vs 5026's human scouting (current results)

315 team-matches in 54 Chezy Champs quals scouted by FRC 5026 (every robot scouted, mostly twice). Reference = the alliance's TBA fuel total split by scouted shots, so it assumes equal accuracy within an alliance.

| | auto-scout | older hand-tuned rule | even 1/3 split | fuel-OPR split |
| --- | ---: | ---: | ---: | ---: |
| team-match fuel error, mean | **39.1** | 46.3 | 81.5 | 57.7 |
| team-match fuel error, median % | **19 %** | 22 % | 33 % | 26 % |
| correlation with the reference | **0.89** | 0.84 | – | 0.78 |
| top scorer of the alliance | **70 %** | 68 % | 33 % | 61 % |
| balls per scoreboard jump given to the robots scouts saw shooting | **66 %** | – | 52 % | 55 % |

* noise floor: the reference built from one scout vs the other differs by 45.9 fuel per team-match on average
* auto-scout uses the learned credit model, which was fitted to these same scouted matches, so this column is in-sample; its cross-validated figures are in "Learned per-team credit split"
