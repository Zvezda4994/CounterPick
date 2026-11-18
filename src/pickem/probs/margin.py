"""Turn bookmaker decimal odds into fair probabilities by removing the margin."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

Method = Literal["power", "proportional"]
METHODS: tuple[Method, ...] = ("power", "proportional")


def implied(odds: ArrayLike) -> NDArray[np.float64]:
    """Raw implied probabilities 1 / odds (they include the bookmaker's margin)."""
    o = np.asarray(odds, dtype=np.float64)
    if o.size == 0 or not np.all(np.isfinite(o)) or np.any(o <= 1.0):
        raise ValueError("decimal odds must be finite numbers greater than 1")
    result: NDArray[np.float64] = 1.0 / o
    return result


def overround(odds: ArrayLike, total: float = 1.0) -> float:
    """Bookmaker margin: how far the implied probabilities exceed what the market should sum to."""
    return float(implied(odds).sum() / total - 1.0)


def power_exponent(pi: NDArray[np.float64], total: float) -> float:
    """The k > 0 with sum(pi ** k) == total. The sum falls as k grows because 0 < pi < 1."""
    if not 0.0 < total < len(pi):
        raise ValueError(f"market total must be between 0 and {len(pi)}, got {total}")
    return float(brentq(lambda k: float(np.sum(pi**k)) - total, 1e-9, 1e4, xtol=1e-14))


def remove_margin(
    odds: ArrayLike, total: float = 1.0, method: Method = "power"
) -> NDArray[np.float64]:
    """Fair probabilities for one market whose outcomes should sum to ``total``.

    ``power`` raises each implied probability to a common exponent, which trims longshots more
    than favourites (the usual favourite-longshot bias); ``proportional`` scales them all alike.
    """
    pi = implied(odds)
    p: NDArray[np.float64]
    if method == "power":
        p = pi ** power_exponent(pi, total)
        return p
    if method == "proportional":
        if not 0.0 < total < len(pi):
            raise ValueError(f"market total must be between 0 and {len(pi)}, got {total}")
        p = pi * (total / pi.sum())
        if np.any(p >= 1.0):
            raise ValueError("proportional normalization gives a probability of 1 or more")
        return p
    raise ValueError(f"unknown margin method {method!r}; use one of {METHODS}")
