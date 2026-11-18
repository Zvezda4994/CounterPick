from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pickem.probs.margin import remove_margin
from pickem.probs.odds_csv import COLUMNS, fair_probabilities, load_odds_csv

EXAMPLES = Path(__file__).parents[2] / "examples"
TEMPLATE = EXAMPLES / "singapore_2026_stage1_odds.csv"
EXAMPLE = EXAMPLES / "example_odds.csv"


def edited(tmp_path: Path, change: Callable[[pd.DataFrame], pd.DataFrame | None]) -> Path:
    df = pd.read_csv(EXAMPLE, comment="#", dtype=str, keep_default_na=False)
    result = change(df)
    path = tmp_path / "odds.csv"
    (df if result is None else result).to_csv(path, index=False)
    return path


def test_template_has_real_stage1_teams_and_blank_odds() -> None:
    sheet = load_odds_csv(TEMPLATE)
    assert sheet.teams[:2] == ("Ninjas in Pyjamas", "PARIVISION")
    assert sheet.teams[12] == "GOSH" and sheet.teams[15] == "Lynn Vision"
    assert sheet.vrs_points[0] == 1564 and sheet.vrs_points[15] == 1267
    assert list(sheet.vrs_points) == sorted(sheet.vrs_points, reverse=True)
    for odds in (sheet.r1_odds, sheet.qualify_odds, sheet.not_qualify_odds, sheet.win_major_odds):
        assert np.isnan(odds).all()
    market = fair_probabilities(sheet)
    assert market.n_r1_pairs == market.n_qualify == market.n_win == 0


def test_example_markets_become_fair_probabilities() -> None:
    sheet = load_odds_csv(EXAMPLE)
    market = fair_probabilities(sheet)
    pairs = market.r1[:8] + market.r1[8:]
    assert pairs == pytest.approx(np.ones(8))
    assert market.qualify.sum() == pytest.approx(8.0)
    assert market.win_share.sum() == pytest.approx(1.0)
    assert market.overrounds["r1"] == pytest.approx([0.06] * 8, abs=0.005)
    assert market.overrounds["qualify"] == pytest.approx([0.081], abs=0.002)
    assert market.warnings == ()


def test_rows_may_come_in_any_order(tmp_path: Path) -> None:
    path = edited(tmp_path, lambda df: df.iloc[::-1])
    assert load_odds_csv(path).teams == load_odds_csv(EXAMPLE).teams


def test_optional_columns_may_be_left_out(tmp_path: Path) -> None:
    path = edited(tmp_path, lambda df: df[["seed", "team", "vrs_points", "r1_odds"]])
    sheet = load_odds_csv(path)
    assert np.isnan(sheet.qualify_odds).all()
    assert fair_probabilities(sheet).n_r1_pairs == 8


def test_yes_no_qualify_prices_are_normalized_per_team(tmp_path: Path) -> None:
    def two_way(df: pd.DataFrame) -> None:
        df.loc[:, "qualify_odds"] = ""
        df.loc[0, "qualify_odds"] = "1.30"
        df.loc[0, "not_qualify_odds"] = "3.40"

    market = fair_probabilities(load_odds_csv(edited(tmp_path, two_way)))
    assert market.n_qualify == 1
    assert market.qualify[0] == pytest.approx(remove_margin([1.30, 3.40])[0])
    assert np.isnan(market.qualify[1:]).all()


def test_overround_warnings(tmp_path: Path) -> None:
    def generous(df: pd.DataFrame) -> None:
        df.loc[0, "r1_odds"] = "2.10"  # 1/2.10 + 1/2.78 < 1: a book below 100%

    market = fair_probabilities(load_odds_csv(edited(tmp_path, generous)))
    assert any("Team 01 v Team 09" in w for w in market.warnings)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda df: df.drop(columns="vrs_points"), "missing columns"),
        (lambda df: df.assign(extra="1"), "unknown columns"),
        (lambda df: df.iloc[:15], "16 team rows"),
        (lambda df: df.assign(seed=["1"] * 16), "each once"),
        (lambda df: df.assign(seed=["x"] * 16), "whole number"),
        (lambda df: df.assign(team=["A"] * 16), "distinct"),
        (lambda df: df.assign(vrs_points=[""] * 16), "vrs_points is blank"),
        (lambda df: df.assign(r1_opponent=["Team 02"] * 16), "plays Team 09"),
        (lambda df: df.assign(r1_odds=["1.0"] * 16), "above 1"),
        (lambda df: df.assign(win_major_odds=["long"] * 16), "not a number"),
    ],
)
def test_invalid_sheets_fail_clearly(
    tmp_path: Path, change: Callable[[pd.DataFrame], pd.DataFrame], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_odds_csv(edited(tmp_path, change))


def test_round1_odds_need_both_teams(tmp_path: Path) -> None:
    def one_sided(df: pd.DataFrame) -> None:
        df.loc[8, "r1_odds"] = ""

    with pytest.raises(ValueError, match="both Team 01 and Team 09"):
        load_odds_csv(edited(tmp_path, one_sided))


def test_partial_qualify_book_needs_the_no_side(tmp_path: Path) -> None:
    def partial(df: pd.DataFrame) -> None:
        df.loc[3, "qualify_odds"] = ""

    with pytest.raises(ValueError, match="all 16 teams"):
        load_odds_csv(edited(tmp_path, partial))

    def no_without_yes(df: pd.DataFrame) -> None:
        df.loc[3, "qualify_odds"] = ""
        df.loc[3, "not_qualify_odds"] = "1.5"

    with pytest.raises(ValueError, match="without qualify_odds"):
        load_odds_csv(edited(tmp_path, no_without_yes))


def test_comments_and_header_spacing(tmp_path: Path) -> None:
    lines = EXAMPLE.read_text(encoding="utf-8").splitlines()
    path = tmp_path / "spaced.csv"
    path.write_text(
        "\n".join(["# a note, with commas", ", ".join(COLUMNS), *lines[3:]]) + "\n",
        encoding="utf-8",
    )
    assert load_odds_csv(path).teams == load_odds_csv(EXAMPLE).teams
