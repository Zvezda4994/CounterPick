import numpy as np
import pytest

from pickem.probs.margin import implied, overround, power_exponent, remove_margin


def test_implied_and_overround() -> None:
    assert implied([2.0, 4.0]).tolist() == [0.5, 0.25]
    assert overround([1.9, 1.9]) == pytest.approx(2 / 1.9 - 1)
    assert overround([2.0] * 16, total=8) == pytest.approx(0.0)


@pytest.mark.parametrize("method", ["power", "proportional"])
def test_two_way_market_sums_to_one(method: str) -> None:
    p = remove_margin([1.43, 2.78], 1.0, method)  # type: ignore[arg-type]
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1]


@pytest.mark.parametrize("method", ["power", "proportional"])
def test_qualify_market_sums_to_eight(method: str) -> None:
    odds = np.linspace(1.3, 3.0, 16)
    p = remove_margin(odds, 8.0, method)  # type: ignore[arg-type]
    assert p.sum() == pytest.approx(8.0)
    assert np.all((p > 0) & (p < 1))


def test_known_values() -> None:
    assert remove_margin([1.9, 1.9]).tolist() == pytest.approx([0.5, 0.5])
    assert remove_margin([1.25, 5.0]).tolist() == pytest.approx([0.8, 0.2])  # no margin
    assert remove_margin([1.5, 2.5], method="proportional").tolist() == pytest.approx(
        [(1 / 1.5) / (1 / 1.5 + 1 / 2.5), (1 / 2.5) / (1 / 1.5 + 1 / 2.5)]
    )


def test_power_method_trims_longshots_more() -> None:
    odds = [1.2, 4.5]
    power = remove_margin(odds, method="power")
    proportional = remove_margin(odds, method="proportional")
    assert power[1] < proportional[1]
    assert power[0] > proportional[0]


def test_power_exponent() -> None:
    pi = implied([1.43, 2.78])
    k = power_exponent(pi, 1.0)
    assert k > 1  # an overround needs shrinking
    assert np.sum(pi**k) == pytest.approx(1.0, abs=1e-12)
    assert power_exponent(implied([2.0, 2.0]), 1.0) == pytest.approx(1.0)


@pytest.mark.parametrize("odds", [[1.0, 2.0], [0.5, 3.0], [np.nan, 2.0], []])
def test_rejects_invalid_odds(odds: list[float]) -> None:
    with pytest.raises(ValueError, match="greater than 1"):
        remove_margin(odds)


def test_rejects_bad_total_and_method() -> None:
    with pytest.raises(ValueError, match="between"):
        remove_margin([2.0, 2.0], total=2.0)
    with pytest.raises(ValueError, match="unknown"):
        remove_margin([2.0, 2.0], method="shin")  # type: ignore[arg-type]
