from __future__ import annotations

import numpy as np
from gymnasium import spaces

from env.config import EnvConfig
from env.observations.builder import ObservationBuilder
from env.state import SimState

ENV_SCHEMA_VERSION = "obs@1694:v5"
AGENT_GRID_WEIGHT = 0.0
AGENT_GRID_RADIUS = 1


class SwarmObserver:
    def __init__(self, config: EnvConfig, *, rng: np.random.Generator | None = None) -> None:
        self.config = config
        self.rng = rng
        self._cache_timestep: int | None = None
        self._cache_obs_by_idx: dict[int, dict[str, np.ndarray]] = {}
        self._cache_obs_map: dict[str, dict[str, np.ndarray]] | None = None
        self._cache_agents: list[str] | None = None

        self.env_schema_version = str(getattr(self.config, "obs_schema_version", ENV_SCHEMA_VERSION))
        if self.env_schema_version != ENV_SCHEMA_VERSION:
            raise ValueError(f"Неизвестная версия env_schema: {self.env_schema_version}")
        grid_width = int(getattr(self.config, "grid_width", 41))
        grid_res = float(getattr(self.config, "grid_res", 1.0))
        if grid_width != 41 or abs(grid_res - 1.0) > 1e-6:
            raise ValueError(
                "obs@1694:v5 требует grid_width=41 и grid_res=1.0; "
                f"получено grid_width={grid_width}, grid_res={grid_res}"
            )
        self.grid_width = int(grid_width)
        self.vector_dim = 2 + 2 + 4 + 2 + 2 + 1

        self.obs_builder = ObservationBuilder(
            self.config.field_size,
            self.config.max_speed,
            self.grid_width,
            self.env_schema_version,
            grid_res=grid_res,
            agent_radius=float(getattr(self.config, "agent_radius", 0.0)),
            agent_grid_weight=AGENT_GRID_WEIGHT,
            agent_blob_radius=AGENT_GRID_RADIUS,
            obs_noise_target=float(getattr(self.config, "obs_noise_target", 0.0)),
            obs_noise_vel=float(getattr(self.config, "obs_noise_vel", 0.0)),
            obs_noise_grid=float(getattr(self.config, "obs_noise_grid", 0.0)),
            rng=rng,
        )

    def sync_from_engine(self, config: EnvConfig) -> None:
        self.obs_builder.field_size = float(config.field_size)
        self.obs_builder.max_speed = float(config.max_speed)
        self.obs_builder.agent_radius = float(getattr(config, "agent_radius", 0.0))
        try:
            self.obs_builder.set_noise(
                obs_noise_target=getattr(config, "obs_noise_target", None),
                obs_noise_vel=getattr(config, "obs_noise_vel", None),
                obs_noise_grid=getattr(config, "obs_noise_grid", None),
            )
        except Exception:
            pass
        self.reset_cache()

    def set_rng(self, rng: np.random.Generator | None) -> None:
        if rng is None:
            return
        self.rng = rng
        try:
            self.obs_builder.set_rng(rng)
        except Exception:
            pass

    def reset_cache(self) -> None:
        """Сбрасывает кэш наблюдений при смене эпизода или параметров."""
        self._cache_timestep = None
        self._cache_obs_by_idx = {}
        self._cache_obs_map = None
        self._cache_agents = None
        try:
            self.obs_builder.reset_cache()
        except Exception:
            pass

    def build(self, state: SimState, idx: int) -> dict[str, np.ndarray]:
        timestep = int(state.timestep)
        if self._cache_timestep != timestep:
            self.reset_cache()
            self._cache_timestep = timestep
        cached = self._cache_obs_by_idx.get(idx)
        if cached is not None:
            return cached
        obs = self.obs_builder.build(state, idx)
        self._cache_obs_by_idx[idx] = obs
        return obs

    def build_all(self, state: SimState, agents: list[str]) -> dict[str, dict[str, np.ndarray]]:
        timestep = int(state.timestep)
        if self._cache_timestep == timestep and self._cache_obs_map is not None and self._cache_agents == agents:
            return self._cache_obs_map
        if self._cache_timestep != timestep:
            self.reset_cache()
            self._cache_timestep = timestep
        obs_map: dict[str, dict[str, np.ndarray]] = {}
        idxs = list(range(len(agents)))
        if hasattr(self.obs_builder, "build_all"):
            obs_list = self.obs_builder.build_all(state, idxs)
            for i, agent in enumerate(agents):
                obs = obs_list[i]
                self._cache_obs_by_idx[i] = obs
                obs_map[agent] = obs
        else:
            for i, agent in enumerate(agents):
                obs = self._cache_obs_by_idx.get(i)
                if obs is None:
                    obs = self.obs_builder.build(state, i)
                    self._cache_obs_by_idx[i] = obs
                obs_map[agent] = obs
        self._cache_obs_map = obs_map
        self._cache_agents = list(agents)
        return obs_map

    def zero_obs(self) -> dict[str, np.ndarray]:
        return {
            "vector": np.zeros((self.vector_dim,), dtype=np.float32),
            "grid": np.zeros((1, self.grid_width, self.grid_width), dtype=np.float32),
        }

    def make_observation_spaces(self, agents: list[str]) -> dict[str, spaces.Dict]:
        return {
            agent: spaces.Dict(
                {
                    "vector": spaces.Box(low=-1.0, high=1.0, shape=(self.vector_dim,), dtype=np.float32),
                    "grid": spaces.Box(
                        low=0.0, high=1.0, shape=(1, self.grid_width, self.grid_width), dtype=np.float32
                    ),
                }
            )
            for agent in agents
        }
