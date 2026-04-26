"""Реестр RNG-потоков для детерминизма по именам подсистем."""

from __future__ import annotations

import zlib

import numpy as np


def _stable_hash(name: str) -> int:
    """Стабильный хэш имени, независимый от Python hash()."""
    return zlib.adler32(name.encode("utf-8")) & 0xFFFFFFFF


class RNGRegistry:
    """Реестр RNG: каждый поток определяется базовым seed и именем."""

    def __init__(self, seed: int | None = None) -> None:
        self._streams: dict[str, np.random.Generator] = {}
        self._seed = self._normalize_seed(seed)

    @property
    def seed(self) -> int:
        return int(self._seed)

    def reset(self, seed: int | None) -> int:
        """Сбрасывает базовый seed и очищает потоки."""
        self._seed = self._normalize_seed(seed)
        self._streams = {}
        return int(self._seed)

    def get(self, name: str) -> np.random.Generator:
        """Возвращает детерминированный RNG для указанного имени."""
        key = str(name)
        rng = self._streams.get(key)
        if rng is None:
            ss = np.random.SeedSequence([int(self._seed), _stable_hash(key)])
            rng = np.random.default_rng(ss)
            self._streams[key] = rng
        return rng

    @staticmethod
    def _normalize_seed(seed: int | None) -> int:
        if seed is None:
            return int(np.random.default_rng().integers(0, 2**32 - 1))
        return int(seed)
