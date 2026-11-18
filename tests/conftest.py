import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded generator so simulation tests are reproducible."""
    return np.random.default_rng(20261117)
