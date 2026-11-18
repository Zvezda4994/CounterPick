import importlib

import pytest

SUBPACKAGES = ["data", "ratings", "probs", "swiss", "optimizer"]


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"pickem.{name}")
    assert module.__doc__
