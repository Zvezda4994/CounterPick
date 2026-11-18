import time

import numpy as np
import pytest
from numpy.typing import NDArray

from pickem.swiss import StageInput, SwissResult, simulate_from_uniforms, simulate_swiss
from pickem.swiss.rules import round_layouts
from pickem.swiss.simulate import _pick_pairing
from tests.swiss.reference import Stage as RefStage
from tests.swiss.reference import _pair_group, run_stage

TEAMS = tuple(f"T{i:02d}" for i in range(16))
TOTAL_MATCHES = 8 + 8 + 8 + 6 + 3
FINAL_RECORD_COUNTS = {(3, 0): 2, (3, 1): 3, (3, 2): 3, (2, 3): 3, (1, 3): 3, (0, 3): 2}


def logistic_stage(
    rng: np.random.Generator, *, spread: float = 1.0, shuffle_seeds: bool = False
) -> StageInput:
    """Random but valid matrices: P(i beats j) = sigmoid(k * (s_i - s_j))."""
    strength = rng.normal(scale=spread, size=16)
    gap = strength[:, None] - strength[None, :]
    seeds = rng.permutation(16) + 1 if shuffle_seeds else np.arange(1, 17)
    return StageInput(
        teams=TEAMS,
        seeds=tuple(int(s) for s in seeds),
        p_bo1=1.0 / (1.0 + np.exp(-gap)),
        p_bo3=1.0 / (1.0 + np.exp(-1.6 * gap)),
    )


def constant_stage(p: float) -> StageInput:
    matrix = np.full((16, 16), p)
    return StageInput(TEAMS, tuple(range(1, 17)), matrix, matrix)


@pytest.fixture(scope="module")
def big_result() -> tuple[StageInput, SwissResult]:
    rng = np.random.default_rng(7)
    stage = logistic_stage(rng, spread=0.8, shuffle_seeds=True)
    return stage, simulate_swiss(stage, 20_000, rng, record_matches=True)


# --- outcome invariants -------------------------------------------------------------------


def test_records_sum_correctly(big_result: tuple[StageInput, SwissResult]) -> None:
    _, res = big_result
    assert (res.wins.sum(axis=1) == TOTAL_MATCHES).all()
    assert (res.losses.sum(axis=1) == TOTAL_MATCHES).all()
    finished_up, finished_down = res.wins == 3, res.losses == 3
    assert (finished_up ^ finished_down).all()  # every team ends exactly one way
    assert ((res.wins + res.losses >= 3) & (res.wins + res.losses <= 5)).all()


def test_exactly_8_advance_with_two_3_0_and_two_0_3(
    big_result: tuple[StageInput, SwissResult],
) -> None:
    _, res = big_result
    assert ((res.wins == 3).sum(axis=1) == 8).all()
    for (w, lo), count in FINAL_RECORD_COUNTS.items():
        assert (((res.wins == w) & (res.losses == lo)).sum(axis=1) == count).all(), (w, lo)


def test_final_seed_orders_by_record(big_result: tuple[StageInput, SwissResult]) -> None:
    _, res = big_result
    assert (np.sort(res.final_seed, axis=1) == np.arange(1, 17)).all()
    assert ((res.final_seed <= 8) == (res.wins == 3)).all()
    order = np.argsort(res.final_seed, axis=1)
    rank = 4 * (3 - np.take_along_axis(res.wins, order, axis=1)) + np.take_along_axis(
        res.losses, order, axis=1
    )
    assert (np.diff(rank.astype(int), axis=1) >= 0).all()


# --- pairing rules, checked directly on the match log --------------------------------------


def test_pairing_rules_respected(big_result: tuple[StageInput, SwissResult]) -> None:
    stage, res = big_result
    log = res.matches
    assert log is not None
    n = res.n_sims
    rows = np.arange(n)[:, None]
    wins = np.zeros((n, 16), dtype=int)
    losses = np.zeros((n, 16), dtype=int)
    played = np.zeros((n, 16, 16), dtype=bool)
    seed_of = np.asarray(stage.seeds)
    no_forced = res.rematches == 0

    for r in range(5):
        m = int((log.best_of[r] > 0).sum())
        assert (log.high[:, r, m:] == -1).all()
        high, low = log.high[:, r, :m].astype(int), log.low[:, r, :m].astype(int)

        # same record going in
        assert (wins[rows, high] == wins[rows, low]).all()
        assert (losses[rows, high] == losses[rows, low]).all()

        # every unfinished team plays exactly once; finished teams sit out
        appearances = np.zeros((n, 16), dtype=int)
        np.add.at(appearances, (np.repeat(np.arange(n), m), high.ravel()), 1)
        np.add.at(appearances, (np.repeat(np.arange(n), m), low.ravel()), 1)
        active = (wins < 3) & (losses < 3)
        assert (appearances == active.astype(int)).all()

        if r == 0:
            assert (seed_of[high] == np.arange(1, 9)).all()
            assert (seed_of[low] == np.arange(9, 17)).all()

        # BO3 exactly when the match can advance or eliminate someone
        stakes = (wins[rows, high] == 2) | (losses[rows, high] == 2)
        assert (stakes == (log.best_of[r, :m] == 3)).all()

        # no rematches unless the simulator reported a forced one
        assert not played[rows, high, low][no_forced].any()

        winner = np.where(log.high_won[:, r, :m], high, low)
        loser = np.where(log.high_won[:, r, :m], low, high)
        wins[rows, winner] += 1
        losses[rows, loser] += 1
        played[rows, high, low] = True
        played[rows, low, high] = True

    assert (wins == res.wins).all()
    assert (losses == res.losses).all()
    difficulty_recount: NDArray[np.int64] = (played * (wins - losses)[:, None, :]).sum(axis=2)
    assert (difficulty_recount == res.difficulty).all()


# --- exact agreement with the scalar reference --------------------------------------------


@pytest.mark.parametrize(("seed", "spread"), [(11, 0.3), (12, 1.0), (13, 3.0)])
def test_matches_scalar_reference(seed: int, spread: float) -> None:
    rng = np.random.default_rng(seed)
    stage = logistic_stage(rng, spread=spread, shuffle_seeds=True)
    u = rng.random((700, 5, 8))
    res = simulate_from_uniforms(stage, u, record_matches=True)
    log = res.matches
    assert log is not None

    by_seed = stage.by_seed  # seed index -> input team index
    p1 = stage.p_bo1[np.ix_(by_seed, by_seed)].tolist()
    p3 = stage.p_bo3[np.ix_(by_seed, by_seed)].tolist()
    column = np.asarray(stage.seeds) - 1  # input team index -> seed index
    for i in range(len(u)):
        ref = run_stage(u[i].tolist(), p1, p3)
        slot = {r: 0 for r in range(5)}
        for match in ref.matches:
            j = slot[match.round]
            slot[match.round] += 1
            got = (
                int(log.high[i, match.round, j]),
                int(log.low[i, match.round, j]),
                bool(log.high_won[i, match.round, j]),
                int(log.best_of[match.round, j]),
            )
            want = (
                int(by_seed[match.high]),
                int(by_seed[match.low]),
                match.high_won,
                match.best_of,
            )
            assert got == want, f"sim {i} round {match.round + 1} slot {j}"
        assert slot == {0: 8, 1: 8, 2: 8, 3: 6, 4: 3}
        final_seed = ref.final_seed()
        assert res.wins[i].tolist() == [ref.wins[s] for s in column]
        assert res.losses[i].tolist() == [ref.losses[s] for s in column]
        assert res.difficulty[i].tolist() == [ref.difficulty(int(s)) for s in column]
        assert res.final_seed[i].tolist() == [final_seed[s] for s in column]
        assert int(res.rematches[i]) == ref.rematches


# --- statistics and deterministic cases ---------------------------------------------------


def test_coin_flips_give_textbook_record_distribution() -> None:
    n = 40_000
    res = simulate_swiss(constant_stage(0.5), n, np.random.default_rng(3))
    expected = {(3, 0): 1 / 8, (3, 1): 3 / 16, (3, 2): 3 / 16, (2, 3): 3 / 16, (1, 3): 3 / 16}
    expected[(0, 3)] = 1 / 8
    for (w, lo), p in expected.items():
        freq = ((res.wins == w) & (res.losses == lo)).mean(axis=0)
        sigma = np.sqrt(p * (1 - p) / n)
        assert (np.abs(freq - p) < 5 * sigma).all(), ((w, lo), freq)


def test_higher_seed_always_wins() -> None:
    better = (np.arange(16)[:, None] < np.arange(16)[None, :]).astype(float)
    stage = StageInput(TEAMS, tuple(range(1, 17)), better, better)
    res = simulate_swiss(stage, 50, np.random.default_rng(0))
    assert (res.wins == res.wins[0]).all() and (res.losses == res.losses[0]).all()
    record = [f"{w}-{lo}" for w, lo in zip(res.wins[0], res.losses[0], strict=True)]
    assert record[0] == record[1] == "3-0"
    assert record[14] == record[15] == "0-3"
    assert (res.wins[0, :8] == 3).all() and (res.wins[0, 8:] < 3).all()
    assert res.final_seed[0, :2].tolist() == [1, 2]


def test_input_team_order_does_not_matter() -> None:
    rng = np.random.default_rng(5)
    stage = logistic_stage(rng, shuffle_seeds=True)
    perm = rng.permutation(16)
    shuffled = StageInput(
        teams=tuple(stage.teams[i] for i in perm),
        seeds=tuple(stage.seeds[i] for i in perm),
        p_bo1=stage.p_bo1[np.ix_(perm, perm)],
        p_bo3=stage.p_bo3[np.ix_(perm, perm)],
    )
    u = rng.random((500, 5, 8))
    a = simulate_from_uniforms(stage, u)
    b = simulate_from_uniforms(shuffled, u)
    for name in ("wins", "losses", "difficulty", "final_seed"):
        assert (getattr(a, name)[:, perm] == getattr(b, name)).all(), name


def test_batches_reproduce_a_single_run() -> None:
    stage = logistic_stage(np.random.default_rng(8))
    one = simulate_swiss(stage, 1_000, np.random.default_rng(9), record_matches=True)
    many = simulate_swiss(
        stage, 1_000, np.random.default_rng(9), record_matches=True, batch_size=137
    )
    for name in ("wins", "losses", "difficulty", "final_seed", "rematches"):
        assert (getattr(one, name) == getattr(many, name)).all(), name
    assert one.matches is not None and many.matches is not None
    assert (one.matches.high == many.matches.high).all()


@pytest.mark.parametrize("n_sims", [100_000, pytest.param(1_000_000, marks=pytest.mark.slow)])
def test_scale(n_sims: int, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(2026)
    stage = logistic_stage(rng, spread=1.0)
    start = time.perf_counter()
    res = simulate_swiss(stage, n_sims, rng)  # 1M runs as 10 batches of 100k
    elapsed = time.perf_counter() - start
    assert res.n_sims == n_sims
    assert ((res.wins == 3).sum(axis=1) == 8).all()
    assert (res.wins.sum(axis=1) == TOTAL_MATCHES).all()
    with capsys.disabled():
        print(
            f"\n{n_sims:,} sims: {elapsed:.2f}s; forced rematches in "
            f"{int((res.rematches > 0).sum())} of {res.n_sims:,} sims"
        )


# --- forced-rematch fallback (never triggered in 100k random sims, so tested directly) -----


@pytest.mark.parametrize(
    ("round_index", "size", "fresh_pairs", "expected_forced"),
    [
        (3, 6, [(0, 3)], 2),  # only 1v4 is new: rows 5, 6, 11 tie on 2 rematches -> row 5
        (3, 6, [], 3),  # every pairing is a rematch -> top row
        (2, 8, [(0, 1), (2, 3)], 2),  # high-low order: last branch for 1v2, then for 3v4
    ],
)
def test_forced_rematch_takes_fewest_then_top_most(
    round_index: int, size: int, fresh_pairs: list[tuple[int, int]], expected_forced: int
) -> None:
    fresh = {frozenset(p) for p in fresh_pairs}
    played = np.zeros((1, 16), dtype=np.int32)
    ref = RefStage()
    for a in range(size):
        for b in range(size):
            if a != b and frozenset((a, b)) not in fresh:
                played[0, a] |= 1 << b
                ref.opponents[a].append(b)

    members = np.arange(size, dtype=np.intp)[None, :]  # standings rank == team id
    group = next(g for g in round_layouts()[round_index].groups if g.size == size)
    pairing, forced = _pick_pairing(members, group.candidates, played)
    ref_pairs = _pair_group(round_index, list(range(size)), ref)

    assert [tuple(p) for p in pairing[0].tolist()] == ref_pairs
    assert int(forced[0]) == ref.rematches == expected_forced
    assert all(frozenset(p) in {frozenset(q) for q in ref_pairs} for p in fresh_pairs)


# --- input validation ---------------------------------------------------------------------


def test_rejects_bad_inputs() -> None:
    good = np.full((16, 16), 0.5)
    seeds = tuple(range(1, 17))
    with pytest.raises(ValueError, match="distinct"):
        StageInput(("A",) * 16, seeds, good, good)
    with pytest.raises(ValueError, match="permutation"):
        StageInput(TEAMS, (1,) * 16, good, good)
    with pytest.raises(ValueError, match="shape"):
        StageInput(TEAMS, seeds, np.full((15, 15), 0.5), good)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        StageInput(TEAMS, seeds, np.full((16, 16), 1.5), good)
    skewed = good.copy()
    skewed[0, 1] = 0.7
    with pytest.raises(ValueError, match="equal 1"):
        StageInput(TEAMS, seeds, good, skewed)
    stage = constant_stage(0.5)
    with pytest.raises(ValueError, match="n_sims"):
        simulate_swiss(stage, 0, np.random.default_rng())
    with pytest.raises(ValueError, match="shape"):
        simulate_from_uniforms(stage, np.zeros((3, 4, 8)))


def test_input_arrays_are_copied_and_read_only() -> None:
    matrix = np.full((16, 16), 0.5)
    stage = StageInput(TEAMS, tuple(range(1, 17)), matrix, matrix)
    matrix[0, 1] = 0.9
    assert stage.p_bo1[0, 1] == 0.5
    with pytest.raises(ValueError):
        stage.p_bo1[0, 1] = 0.1
