from pathlib import Path

import numpy as np
import pytest

from pickem.data.stage_file import StageFile, load_stage
from pickem.fantasy.lineup import Evaluator
from pickem.fantasy.outcomes import MapOutcomes, simulate_map_outcomes
from pickem.fantasy.performance import PlayerSims, simulate_players
from pickem.fantasy.players import CARD_STATS, PlayerPool, load_players

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_STAGE = ROOT / "examples" / "example_stage.json"


def write_players(path: Path, teams: tuple[str, ...], *, cards: bool = True) -> Path:
    """Five players per team; seed-1 players are pricier and better, one AWPer per team."""
    header = ["team", "hltv_rank", "player", "price", *(CARD_STATS if cards else ())]
    lines = [",".join(header)]
    for t, team in enumerate(teams):
        for k in range(5):
            price = 230_000 - 3_000 * t - 8_000 * k
            row = [team, str(10 + 3 * t), f"P{t:02d}{k}", str(price)]
            if cards:
                rating = 1.20 - 0.01 * t - 0.04 * k
                awp = 0.36 if k == 1 else 0.01
                row += [
                    f"{rating:.2f}",
                    f"{rating + 0.03:.2f}",
                    f"{rating - 0.03:.2f}",
                    f"{awp:.2f}",
                    f"{40 if k == 1 else 56 + k}",
                    f"{14 - k}",
                    "1.3",
                    f"{18 + 2 * k}",
                    f"{19 - k}",
                    f"{0.60 + 0.02 * k:.2f}",
                ]
            lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="session")
def stage_file() -> StageFile:
    return load_stage(EXAMPLE_STAGE)


@pytest.fixture(scope="session")
def pool(stage_file: StageFile, tmp_path_factory: pytest.TempPathFactory) -> PlayerPool:
    path = write_players(tmp_path_factory.mktemp("pool") / "players.csv", stage_file.stage.teams)
    return load_players(path, stage_file.stage.teams)


@pytest.fixture(scope="session")
def outcomes(stage_file: StageFile) -> MapOutcomes:
    return simulate_map_outcomes(stage_file, 3000, np.random.default_rng(11))


@pytest.fixture(scope="session")
def sims(outcomes: MapOutcomes, pool: PlayerPool) -> PlayerSims:
    return simulate_players(outcomes, pool, np.random.default_rng(12))


@pytest.fixture(scope="session")
def evaluator(sims: PlayerSims) -> Evaluator:
    return Evaluator(sims)
