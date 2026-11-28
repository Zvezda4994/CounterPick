## Card leaderboard: singapore_stage2

Scored before lock on `snapshots/singapore_stage2.json` (frozen 2026-11-28T02:20:21+00:00, sha256 8f8a3d01d95a), 100,000 sims with strength uncertainty (spread 1.00), seed 1, code cdd8ea4. Optimal card: Optimizer #1.

| # | Card | Source | Approach | Overlap | P(≥5) | P(≥6) | E[correct] | Correct | Cleared |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Mine (submitted) | me | optimizer pick | 10/10 | 41.0% | 18.1% | 4.19 | – | – |
| 2 | Optimizer #1 | optimizer | optimizer pick | 10/10 | 41.0% | 18.1% | 4.19 | – | – |
| 3 | Seed chalk | seed chalk | seed chalk | 5/10 | 25.8% | 9.1% | 3.64 | – | – |

Results are not in yet.

### Teams: model vs actual

| Seed | Team | P(3-0) | P(3-1/3-2) | P(0-3) | Actual | Model P(actual record) |
|---|---|---|---|---|---|---|
| 1 | FUT | 33.3% | 49.9% | 1.8% | – | – |
| 2 | Nemiga | 19.2% | 50.4% | 4.6% | – | – |
| 3 | G2 | 16.3% | 47.6% | 5.6% | – | – |
| 4 | 1win | 12.3% | 42.1% | 7.5% | – | – |
| 5 | Natus Vincere | 35.1% | 54.4% | 1.4% | – | – |
| 6 | Astralis | 6.9% | 28.1% | 12.5% | – | – |
| 7 | The MongolZ | 17.9% | 46.3% | 4.9% | – | – |
| 8 | MIBR | 9.8% | 41.0% | 9.0% | – | – |
| 9 | FlyQuest | 2.4% | 16.6% | 27.6% | – | – |
| 10 | Ninjas in Pyjamas | 11.1% | 45.5% | 8.5% | – | – |
| 11 | Liquid | 8.2% | 37.9% | 11.8% | – | – |
| 12 | Eternal Fire | 6.2% | 31.1% | 15.1% | – | – |
| 13 | PARIVISION | 10.8% | 50.8% | 8.1% | – | – |
| 14 | Imperial | 1.8% | 11.6% | 30.1% | – | – |
| 15 | Lynn Vision | 1.3% | 9.4% | 38.2% | – | – |
| 16 | 100 Thieves | 7.5% | 37.1% | 13.3% | – | – |

### Cards

#### Mine (submitted) (me)

**Approach:** optimizer pick. **Overlap with the optimal card:** 10/10 identical.

| Slot | Team | P(correct) | Result | Model P(that result) |
|---|---|---|---|---|
| 3-0 | FUT | 33.3% | – | – |
| 3-0 | Natus Vincere | 35.1% | – | – |
| 0-3 | Imperial | 30.1% | – | – |
| 0-3 | Lynn Vision | 38.2% | – | – |
| Advance | Nemiga | 50.4% | – | – |
| Advance | G2 | 47.6% | – | – |
| Advance | 1win | 42.1% | – | – |
| Advance | The MongolZ | 46.3% | – | – |
| Advance | Ninjas in Pyjamas | 45.5% | – | – |
| Advance | PARIVISION | 50.8% | – | – |

**Identical to the optimal card.**

**Predicted:** P(≥5) 41.0% ± 0.16, P(≥6) 18.1%, E[correct] 4.19.

#### Optimizer #1 (optimizer)

**Approach:** optimizer pick. **Overlap with the optimal card:** 10/10 identical.

| Slot | Team | P(correct) | Result | Model P(that result) |
|---|---|---|---|---|
| 3-0 | FUT | 33.3% | – | – |
| 3-0 | Natus Vincere | 35.1% | – | – |
| 0-3 | Imperial | 30.1% | – | – |
| 0-3 | Lynn Vision | 38.2% | – | – |
| Advance | Nemiga | 50.4% | – | – |
| Advance | G2 | 47.6% | – | – |
| Advance | 1win | 42.1% | – | – |
| Advance | The MongolZ | 46.3% | – | – |
| Advance | Ninjas in Pyjamas | 45.5% | – | – |
| Advance | PARIVISION | 50.8% | – | – |

**Identical to the optimal card.**

**Predicted:** P(≥5) 41.0% ± 0.16, P(≥6) 18.1%, E[correct] 4.19.

#### Seed chalk (seed chalk)

**Approach:** seed chalk. **Overlap with the optimal card:** 5/10 identical, plus 2 team(s) in a different slot.

| Slot | Team | P(correct) | Result | Model P(that result) |
|---|---|---|---|---|
| 3-0 | FUT | 33.3% | – | – |
| 3-0 | Nemiga | 19.2% | – | – |
| 0-3 | Lynn Vision | 38.2% | – | – |
| 0-3 | 100 Thieves | 13.3% | – | – |
| Advance | G2 | 47.6% | – | – |
| Advance | 1win | 42.1% | – | – |
| Advance | Natus Vincere | 54.4% | – | – |
| Advance | Astralis | 28.1% | – | – |
| Advance | The MongolZ | 46.3% | – | – |
| Advance | MIBR | 41.0% | – | – |

**Differences from the optimal card**, fixed one at a time. Each step makes the change that leaves the best card at that point; because picks interact, a later step can gain more than an earlier one. The gains add up to the full gap of +15.2 points of P(≥5):
1. Advance: Astralis → PARIVISION: +5.7 points
2. 0-3: 100 Thieves → Imperial: +4.8 points
3. 3-0: Nemiga → Natus Vincere (Natus Vincere leaves Advance, Nemiga takes its place): +3.2 points
4. Advance: MIBR → Ninjas in Pyjamas: +1.6 points

**Predicted:** P(≥5) 25.8% ± 0.14, P(≥6) 9.1%, E[correct] 3.64.
