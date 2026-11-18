# CounterPick

What's the best way to get a Diamond coin?

A calculator, analyst picks and gut feel have gotten me gold.

CounterPick is a Pick'Em optimizer for CS2 Majors. It turns betting odds into team strengths, simulates each Swiss stage 100,000 times under Valve's rulebook, and picks the card most likely to get at least 5 of 10 right. I built it for the PGL Singapore Major (Nov–Dec 2026).

## The scoring is weirder than it looks

Each Swiss stage card has 10 picks: two teams to go 3-0, two to go 0-3, and six to advance at 3-1 or 3-2. You need 5.

If you put a team in an advance slot and they go 3-0, that's a miss. Only two teams can go 3-0, so when someone you didn't pick does it, you usually lose a 3-0 pick and an advance pick off the same result. And 3-0 odds depend a lot on who you play first. Round 1 is seed k vs seed k+8, so an easy opener counts for almost as much as being good.

The coin also doesn't care whether you got 5 or 8. So CounterPick maximizes P(correct ≥ 5), not expected correct picks. Those usually point to similar cards, but not always the same one.

## How it works

```
odds CSV → strength fit → Swiss simulator → card optimizer → card
```

### Strength fit

Bookmakers already price in form, roster changes and map pools, so I start from their odds instead of building a rating model from scratch. Round-1 odds only tell you how each pair compares, not how the pairs compare to each other, so the fit also uses stage qualify odds and outright winner odds, with Valve's VRS points as a weak prior. It's a joint Bradley-Terry model. The bookmaker margin is removed with the power method (winner odds are only used as each team's share within the stage).

The fit isn't certain about every team, and the simulator knows that. Each batch of sims draws team strengths from the fit's covariance, so the results include "the odds might be off" as well as normal match randomness.

### Swiss simulator

This follows the Major rulebook line by line: round-1 seeding, Buchholz difficulty, the pairing priority table for rounds 4 and 5, rematch avoidance, BO1 vs BO3 depending on stakes. It's vectorized in NumPy and runs 100k stages in about 1.1 seconds.

### Optimizer

There are 10,090,080 legal cards. The search builds one greedily, then does local search with swaps inside a category and exchanges between categories, from several starting cards. Moves are scored in batches with matrix-vector products.

The best card always looks a little better on the sims it was picked from, so the top cards get re-scored on a separate holdout batch. Those are the numbers I report.

## Checking that it's right

I replayed all three Swiss stages of the Austin 2025 Major. Given the real seeds and results, the simulator reproduces every pairing across all 15 rounds, every Buchholz score Valve published, and the seeds each stage handed to the next.

For the optimizer, a separate brute-force script scores all 10,090,080 cards on small test stages. The optimizer's top 5 matches it exactly, tiebreaks included.

The odds fit was checked by generating odds from strengths I chose, adding bookmaker margins (5% on match odds, 8% on qualify odds), and fitting them back. Every strength comes back within 0.05 logit.

There's also a slow, plain version of the Swiss rules that the fast one is compared against on the same random draws. 222 tests in total, with strict mypy and ruff.

## Singapore Stage 1: my card vs the optimizer

| | Mine | Optimizer |
|---|---|---|
| 3-0 | 9z, NiP | PARIVISION, 100 Thieves |
| 0-3 | Lynn Vision, Chinggis Warriors | Lynn Vision, Chinggis Warriors |
| Advance | 100T, PARIVISION, FaZe, HEROIC, M80, Liquid | NiP, HEROIC, FaZe, Liquid, M80, GOSH |
| P(≥5) | 40.1% | 45.3% |
| E[correct] | 4.21 | 4.34 |

Expected correct is about the same. P(≥5) isn't, and nearly all of the difference is the 3-0 picks.

I had 9z going 3-0 because of their recent upsets. They do it in 13% of sims. 100 Thieves open against the weakest team in the stage and do it in 27%. Meanwhile I'd put 100 Thieves and PARIVISION in advance slots, where a 3-0 run would have cost me a pick.

The best card is still only 45%. Stage 1 is mostly BO1 (only matches that decide advancement or elimination are BO3) between teams that are close in level, and there isn't a card that fixes that.

## Results

| Stage | Card P(≥5) | Correct | Cleared |
|---|---|---|---|
| Stage 1 | 45.3% | | |
| Stage 2 | | | |
| Stage 3 | | | |
| Playoffs | | | |

I'll fill this in as the event goes.

## Card leaderboard

Every card I can find (analysts, friends, mine, the optimizer, seed chalk) is scored before lock on a frozen snapshot of the inputs, then checked against what actually happened.

### Singapore Stage 1

<!-- leaderboard:singapore_stage1 -->
Filled in by `pickem-cards render --name singapore_stage1 --readme README.md`.
<!-- /leaderboard:singapore_stage1 -->

## Usage

```bash
uv sync

cp examples/singapore_2026_stage1_odds.csv data/stage1_odds.csv   # then fill in odds
uv run pickem-odds data/stage1_odds.csv --out data/stage1.json
uv run pickem-optimize data/stage1.json --compare-spread

# score any card against the best one (this is "Mine" from the table above)
uv run pickem-optimize data/stage1.json --card "3-0: 9z, NiP; 0-3: Lynn Vision, Chinggis Warriors; advance: 100 Thieves, PARIVISION, FaZe, HEROIC, M80, Liquid"
```

`pickem-next-stage` builds the next stage's odds sheet from the previous stage's results and Valve's VRS file.

`pickem-cards` runs the card leaderboard: `freeze` the stage file at lock, `score` the cards in `cards/<stage>.csv`, add `results` after the stage, and `render` the table into this README.

## Roadmap

- [x] Swiss simulator
- [x] P(≥5) optimizer with holdout scoring
- [x] Odds layer with strength uncertainty
- [x] Next-stage builder
- [ ] Playoff picks
- [ ] My own rating model (map-level Glicko-2), backtested against the market on Austin, Budapest and Cologne

## Credits

Rules from [ValveSoftware/counter-strike_rules_and_regs](https://github.com/ValveSoftware/counter-strike_rules_and_regs). Seeds and VRS points from [ValveSoftware/counter-strike_regional_standings](https://github.com/ValveSoftware/counter-strike_regional_standings). Match data from [Liquipedia](https://liquipedia.net/counterstrike/) (CC BY-SA). Built with [Claude Code](https://claude.com/claude-code).
