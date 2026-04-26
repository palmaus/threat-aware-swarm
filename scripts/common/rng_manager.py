from __future__ import annotations

import os
import random

import numpy as np

try:  # Опциональная зависимость.
    import torch
except Exception:  # pragma: no cover - опциональная
    torch = None

from env.rng_registry import _stable_hash


class SeedManager:
    """Иерархический менеджер сидов для детерминизма по именам подсистем."""

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    def child_seed(self, name: str) -> int:
        """Возвращает дочерний seed, стабильный по имени."""
        key = str(name)
        ss = np.random.SeedSequence([self.seed, _stable_hash(key)])
        return int(ss.generate_state(1, dtype=np.uint32)[0])

    def rng(self, name: str) -> np.random.Generator:
        """Возвращает отдельный RNG‑поток для подсистемы."""
        return np.random.default_rng(self.child_seed(name))

    def seed_python(self) -> None:
        """Фиксирует RNG Python для воспроизводимости."""
        random.seed(self.seed)

    def seed_numpy(self) -> None:
        """Фиксирует глобальный RNG NumPy."""
        np.random.seed(self.seed)

    def seed_torch(self) -> None:
        """Фиксирует RNG Torch (если доступен)."""
        if torch is None:
            return
        seed = self.child_seed("torch")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def seed_all(self) -> None:
        """Фиксирует базовые генераторы и PYTHONHASHSEED."""
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        self.seed_python()
        self.seed_numpy()
        self.seed_torch()
