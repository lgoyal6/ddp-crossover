from __future__ import annotations

import numpy as np

from verify_restart_equivalence import compare


def test_compare_randomstate_by_saved_state() -> None:
    left = np.random.RandomState(17)
    right = np.random.RandomState(17)
    failures: list[str] = []

    compare(left, right, "rng", failures=failures)

    assert failures == []


def test_compare_detects_randomstate_difference() -> None:
    left = np.random.RandomState(17)
    right = np.random.RandomState(29)
    failures: list[str] = []

    compare(left, right, "rng", failures=failures)

    assert failures == ["rng[1]: array mismatch"]
