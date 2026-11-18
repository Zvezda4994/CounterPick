from pathlib import Path

import pytest

from pickem.data.vrs import load_vrs, parse_vrs

EXCERPT = Path(__file__).parents[1] / "fixtures" / "vrs_global_2025_05_05_excerpt.md"


def test_reads_valves_table() -> None:
    table = load_vrs(EXCERPT)
    assert table.as_of == "2025-05-05"
    assert len(table.entries) == 33
    falcons = table.lookup("falcons")
    assert (falcons.standing, falcons.points, falcons.team) == (3, 1927, "Falcons")


def test_duplicate_names_use_the_best_standing() -> None:
    table = load_vrs(EXCERPT)
    assert [e.standing for e in table.matches("HEROIC")] == [15, 73]
    assert table.lookup("HEROIC").roster.startswith("LNZ")


def test_standing_not_points_breaks_ties() -> None:
    table = load_vrs(EXCERPT)
    m80, nemiga = table.lookup("M80"), table.lookup("Nemiga")
    assert m80.points == nemiga.points == 1060
    assert (m80.standing, nemiga.standing) == (34, 35)


def test_errors() -> None:
    with pytest.raises(ValueError, match="not in the VRS table"):
        load_vrs(EXCERPT).lookup("Astralis")
    with pytest.raises(ValueError, match="no VRS table rows"):
        parse_vrs("# nothing here\n", "empty.md")
    with pytest.raises(ValueError, match="cannot read VRS row"):
        parse_vrs("| 1 | lots | Team |\n", "bad.md")
