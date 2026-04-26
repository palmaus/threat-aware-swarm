from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class WindField:
    """Абстрактный интерфейс ветра."""

    def reset(self, seed: int | None = None) -> None:  # pragma: no cover - интерфейс
        return

    def step(self, dt: float) -> None:  # pragma: no cover - интерфейс
        return

    def sample(self, positions: np.ndarray) -> np.ndarray:  # pragma: no cover - интерфейс
        raise NotImplementedError


class GlobalOUWind(WindField):
    """Глобальный ветер как OU-процесс (одинаковый для всех позиций)."""

    def __init__(
        self,
        *,
        theta: float = 0.15,
        sigma: float = 0.3,
        mean: tuple[float, float] = (0.0, 0.0),
        rng: np.random.Generator | None = None,
    ) -> None:
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.mean = np.asarray(mean, dtype=np.float32)
        self._rng = rng if rng is not None else np.random.default_rng()
        self._value = np.zeros((2,), dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self._value = np.zeros((2,), dtype=np.float32)

    def step(self, dt: float) -> None:
        dt = float(max(dt, 1e-6))
        noise = self._rng.normal(0.0, 1.0, size=2).astype(np.float32)
        drift = self.theta * (self.mean - self._value) * dt
        diffusion = self.sigma * math.sqrt(dt) * noise
        self._value = (self._value + drift + diffusion).astype(np.float32)

    def sample(self, positions: np.ndarray) -> np.ndarray:
        pos = np.asarray(positions, dtype=np.float32)
        if pos.ndim == 1:
            return self._value.copy()
        return np.broadcast_to(self._value, pos.shape).astype(np.float32)
