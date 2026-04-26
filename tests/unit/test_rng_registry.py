"""Проверка детерминированных RNG-потоков."""

import numpy as np

from env.rng_registry import RNGRegistry


def test_rng_registry_deterministic_by_name():
    reg = RNGRegistry(seed=123)
    a1 = reg.get("a").normal(size=4)
    b1 = reg.get("b").normal(size=4)

    reg.reset(123)
    a2 = reg.get("a").normal(size=4)
    b2 = reg.get("b").normal(size=4)

    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)


def test_rng_registry_order_independent():
    reg1 = RNGRegistry(seed=7)
    a1 = reg1.get("a").normal(size=4)
    b1 = reg1.get("b").normal(size=4)

    reg2 = RNGRegistry(seed=7)
    b2 = reg2.get("b").normal(size=4)
    a2 = reg2.get("a").normal(size=4)

    assert np.allclose(a1, a2)
    assert np.allclose(b1, b2)
