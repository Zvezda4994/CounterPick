import re
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from pickem.optimizer.cli import main

EXAMPLE = Path(__file__).parents[2] / "examples" / "example_stage.json"
FAST = ["--sims", "3000", "--restarts", "3"]


def run(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main([str(a) for a in args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_report_has_every_section(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, err = run(capsys, str(EXAMPLE), *FAST, "--eval-sims", "2000")
    assert code == 0, err
    assert out.startswith("Pick'Em optimizer: Example stage")
    assert "Search: 3,000 sims" in out
    assert "Holdout: 2,000 independent sims" in out
    assert "Team marginals" in out and "P(3-1/3-2)" in out
    assert "Candidates after pruning" in out
    assert "Top 5 cards" in out
    assert len(re.findall(r"^#\d", out, flags=re.MULTILINE)) == 5
    assert out.count("Advance:") == 6  # 5 cards + favorites
    assert "Favorites baseline" in out
    assert "Best card vs favorites" in out
    assert "[search P(>=5)" in out


def test_marginals_listed_in_seed_order(capsys: pytest.CaptureFixture[str]) -> None:
    _, out, _ = run(capsys, str(EXAMPLE), *FAST)
    rows = re.findall(r"^\s+(\d+)\s+Team (\d\d)", out, flags=re.MULTILINE)
    assert [int(seed) for seed, _ in rows] == list(range(1, 17))


def test_in_sample_mode(capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = run(capsys, str(EXAMPLE), *FAST, "--eval-sims", "0", "--top", "2")
    assert code == 0
    assert "in-sample" in out
    assert "Top 2 cards" in out and "[search" not in out


def test_same_seed_same_report(capsys: pytest.CaptureFixture[str]) -> None:
    def body(out: str) -> str:
        return "\n".join(line for line in out.splitlines() if not line.startswith("Search:"))

    _, first, _ = run(capsys, str(EXAMPLE), *FAST, "--seed", "4")
    _, second, _ = run(capsys, str(EXAMPLE), *FAST, "--seed", "4")
    assert body(first) == body(second)


def test_bad_input_file(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    code, _, err = run(capsys, str(tmp_path / "missing.json"))
    assert code == 2 and "error" in err
    bad = tmp_path / "bad.json"
    bad.write_text('{"teams": []}', encoding="utf-8")
    code, _, err = run(capsys, str(bad))
    assert code == 2 and "missing p_bo1" in err


def test_bad_prune(capsys: pytest.CaptureFixture[str]) -> None:
    code, _, err = run(capsys, str(EXAMPLE), *FAST, "--prune", "2")
    assert code == 2 and "prune" in err


def test_console_script_is_registered() -> None:
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts.get("pickem-optimize") == "pickem.optimizer.cli:main"
