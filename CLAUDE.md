# CS2 Major Pick'Em Optimizer
Goal: maximize P(>=5 correct picks) per Swiss stage of the PGL Singapore Major 2026, not expected accuracy.

## Pick'Em scoring (verify against Valve's rulebook: github.com/ValveSoftware/counter-strike_rules_and_regs)
- 10 picks per Swiss stage: 2 teams to go 3-0, 2 teams to go 0-3, 6 teams to advance.
- A 3-0 pick scores only on a 3-0 record; a 0-3 pick only on 0-3.
- An advance pick scores only on 3-1 or 3-2 (a 3-0 team is a miss).
- A team can be in only one category.

## Format
- 3 Swiss stages of 16 teams (top 8 advance), then an 8-team playoff.
- Swiss pairing, BO1/BO3 rules and Buchholz tiebreaks must match Valve's current rulebook exactly.

## Swiss rule interpretations (src/pickem/swiss/rules.py)
- Pinned to rulebook commit 621b7d9 (major-supplemental-rulebook.md L164-L272); Swiss text unchanged since 2025-01-16. If upstream changes, update the pin and rerun tests/swiss/test_austin_2025.py.
- Pair within each W-L group by mid-stage seed: record, then Difficulty (Buchholz, opponents' current W-L) desc, then initial seed.
- Rounds 2-3 "lowest available non-rematch": greedy with lookahead (first rematch-free matching in high-low order).
- Rounds 4-5: first rematch-free row of the 15-row table. If no rematch-free pairing exists: fewest rematches, top-most.
- Validated by replaying all 3 Austin 2025 Swiss stages (pairings, Valve's published Difficulty, next-stage seeds).

## Optimizer (src/pickem/optimizer/)
- Pick'Em scoring is not in Valve's rules repo; it follows the spec above. Re-check the in-game rules before each Major.
- Cards are ranked by exact counts: (#sims >= 5 correct, #sims >= 6, total correct).
- Search: prune 3-0/0-3 candidates by marginal, greedy on E[min(correct, 5)], then steepest-ascent replace/exchange moves with restarts. Verified against brute force over all 10,090,080 cards (tests/optimizer/brute_force.py).
- Report holdout numbers: the best card is chosen on search sims and re-scored on independent sims (pickem-optimize --eval-sims).
- Card leaderboard (pickem-cards): snapshots/ is frozen at lock and never replaced; leaderboards/*.json scores are computed once per card (seed 1) and never rewritten without --rescore. cards/ snapshots/ leaderboards/ are tracked and the repo is public.

## Odds layer (src/pickem/probs/)
- Input: one CSV row per team (examples/singapore_2026_stage1_odds.csv is the template; fill a copy in data/). Decimal odds only; blanks are left out of the fit.
- Margin removal: power method by default; the qualify book sums to 8; win-Major odds are used as within-sheet shares.
- Strengths s are logit per BO1 map; BO3 = p^2(3-2p). Joint MAP fit: round-1 odds + qualify odds (checked against the real Swiss sim) + win-Major shares, VRS prior (0.4 +/- 0.25 logit per 100 points, centered below Valve's Glicko scale).
- Uncertainty: Laplace covariance; pickem-optimize redraws strengths every sims_per_draw sims, scaled by --spread (0 = fixed matrices). --compare-spread shows the effect.
- Singapore 2026 Stage 1 seeds = Valve global VRS of 2026-11-02 (verified against Liquipedia's round-1 pairs).
- Next stage: pickem-next-stage replays the previous stage's results (CSV or --liquipedia) through the Swiss engine (swiss/replay.py); advancing teams = seeds 9-16 by final mid-stage seed, invited = seeds 1-8 by VRS standing. Verified on Austin 2025 Stage 1->2->3.

## Playoffs (src/pickem/playoffs/)
- Seeds = final Stage 3 mid-stage seeds (rulebook L253-L254); QF 1v8, 4v5 (half A) and 2v7, 3v6 (half B).
- Format is a setting: "singapore" (default; BO5 final + BO3 third-place match, per Liquipedia PGL/2026/Singapore/Playoffs) vs "rulebook" (all BO3, no third-place match). BO5 = p^3(1+3q+6q^2).
- All 256 bracket outcomes are enumerated exactly; strength uncertainty = average over draws (search draws vs report draws). Brute force: 128 bracket-consistent picks, 2,048 independent per-round picks. Default objective: >=2 QF AND >=1 SF AND GF correct. Which pick rule applies in-game is still to be confirmed.

## HLTV fantasy (src/pickem/fantasy/, pickem-fantasy)
- Rules are from HLTV's game page (fantasy/676, 2026-11-21). Re-check them before each stage's draft. Each Major stage is its own fantasy game.
- Rating points = 50 * (match Rating 3.0 - 1), confirmed on 7/7 IEM Beijing 2026 player-matches. Team points +6/-3 per match.
- Padding is the truncated mean of played rounds (rating, team, role; boosters are not padded). Confirmed on game 671.
- Budget $1,000,000 and at most 2 players per team: confirmed in the Stage 1 draft UI ("Max 2 players from each team").
- Fantasy rounds = Swiss rounds 1-5, and eliminated teams get no padding. Both were confirmed on Cologne Major 2026 Stage 1 (game 629): 3-0 teams were padded in rounds 4-5, and the 0-3 teams' totals match no padding.
- Roles must differ within a lineup. Confirmed on the Stage 1 team page: a role already held by a teammate is greyed out ("role already assigned"). Roles are set once per player on the team page, with no per-round selector. Whether they can be changed between rounds is not yet known.
- Gaimin Gladiators (game 629) scored 55 points better than 50 * (rating - 1) predicts. Possible causes: very low ratings are floored, or HLTV revised its ratings after June. Unresolved.
- Player model is fitted on 740 IEM Beijing player-maps (HLTV match pages vs card ratings). Booster and Eco Friendly rates are population defaults (marked POPULATION in performance.py).
- HLTV's own trigger rates appear on a signed-in team page, in the booster and role pickers, but only for that team's 5 players.
  - Booster ATRs: data/calibration/hltv_fantasy_676_booster_atr.csv. `--booster-rates` thins or tops up the model's booster successes to match them (fantasy/boosters.py, 2.47 maps per historical match).
  - Role big/small rates: data/calibration/hltv_fantasy_676_role_rates.csv. These are not used by the model yet.
- Card ratings are split into the team's average card and the player's deviation from it:
  - deviation: 0.81 +/- 0.13; team average: 0.29 +/- 0.37 once the map result is known.
  - By default team strength comes only from the odds, through the simulated results. This avoids tier-2 schedule inflation (e.g. FlyQuest's 1.30 cards).
  - --team-card-weight 1 restores the IEM-fitted team term.
- Card stats come from the draft page (sign-in needed). A player without a card gets price-based estimates and the CLI warns.
- High-upside objective = E[season points] (percentile tiers 1/5/10/.../80% = 25/18/15/12/10/8/6/4/2/1) against a sampled field. Ownership comes from picked_pct where known; otherwise from a model fitted on game 671.
- Lineups are chosen on search sims and reported on holdout sims. The top-K additive search is checked against brute force.

## Stack
Python 3.11+, numpy, pandas, pytest. Vectorized simulation. Typed and tested.

## Structure
src/pickem/ (probs/, swiss/, optimizer/, playoffs/, fantasy/, ratings/, data/), tests/, notebooks/ (backtests), data/ (gitignored raw)
