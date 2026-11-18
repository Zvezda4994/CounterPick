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

## Stack
Python 3.11+, numpy, pandas, pytest. Vectorized simulation. Typed and tested.

## Structure
src/pickem/ (probs/, swiss/, optimizer/, ratings/, data/), tests/, notebooks/ (backtests), data/ (gitignored raw)
