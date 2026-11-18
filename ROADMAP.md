# Roadmap

## Event schedule (PGL Singapore Major 2026)
- Stage 1 picks lock ~Nov 24.
- Playoff picks lock ~Dec 10.
- Rerun the optimizer before each stage locks.

## Phase 1: Swiss simulator (done)
Rulebook-exact, vectorized 16-team Swiss stage simulation (`src/pickem/swiss/`).

## Phase 2: Optimizer (done)
Maximize P(>=5 correct) per stage, with holdout sims and a brute-force check (`src/pickem/optimizer/`, `pickem-optimize`).

## Phase 3: Odds layer (done)
`src/pickem/probs/`, `pickem-odds`; template in `examples/singapore_2026_stage1_odds.csv`.
- Input: CSV of round 1 match odds plus outright odds (win the Major / qualify from the stage).
- VRS points as a weak prior.
- Joint Bradley-Terry fit, since round 1 pairs are disjoint.
- Sample strength uncertainty per simulation batch.
- Output a stage JSON for `pickem-optimize`.
- Ship a CSV template.

## Phase 3b: Playoffs (built; one open question)
`src/pickem/playoffs/`, `pickem-playoffs`.
- 8-team bracket sim: BO3, BO5 grand final, third-place decider. (done)
- Brute-force all 128 brackets, plus all 2,048 independent per-round picks. (done)
- Maximize P(>=2 QF AND >=1 SF AND GF correct). (done)
- Confirm in-game whether picks must follow the bracket. (open)

## Phase 4: Portfolio
- Data pipeline (bo3.gg / Liquipedia / PandaScore) with cross-source validation.
- Map-level Glicko-2.
- Backtest on Austin / Budapest / Cologne.
- Compare market vs model vs blend.

## Phase 5: README with results
