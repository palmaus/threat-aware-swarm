"""Контроллер интерфейса, соединяющий среду, политику и рендерер."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from env.config import EnvConfig
from common.policy.obs_schema import OBS_VECTOR_DIM, fields_to_obs_vector
from common.runtime.episode_runner import policy_actions as _shared_policy_actions
from common.runtime.episode_runner import reset_policy_context
from common.runtime.path_utils import resolve_repo_path
from common.runtime.env_factory import make_pz_env as _common_make_pz_env
from ui.config import UIConfig
from ui.overlay import OverlayFlags
from ui.policy_workers import build_policy_pool
from ui.policies import PPO_AVAILABLE, create_policy
from ui.scenes import load_scenes_from_roots
from ui.telemetry import build_payload
from ui.telemetry_dto import TelemetryPayload
from ui.tunables import apply_tunables as apply_runtime_tunables
from ui.tunables import collect_tunables

logger = logging.getLogger(__name__)


@dataclass
class ControllerState:
    policy: str
    scene: str
    seed: int
    n_agents: int
    model_path: str
    deterministic: bool
    overlay: dict
    tunables: dict
    oracle: dict


def _policy_actions(env: Any, policy, obs_map: dict | None, infos_map: dict | None) -> dict:
    return _shared_policy_actions(env, policy, obs_map, infos_map)


class SwarmController:
    def __init__(
        self,
        ui_cfg: UIConfig,
        env_cfg: EnvConfig | None = None,
        *,
        scene_root=None,
        user_scene_root=None,
    ):
        self.ui_cfg = ui_cfg
        self.cfg = env_cfg or EnvConfig()
        self.env = _common_make_pz_env(
            max_steps=ui_cfg.max_steps,
            goal_radius=ui_cfg.goal_radius,
            config=self.cfg,
            oracle_enabled=bool(getattr(ui_cfg, "oracle_enabled", False)),
            oracle_async=bool(getattr(ui_cfg, "oracle_async", True)),
            reset=False,
        )
        try:
            self.env.config.oracle_update_interval = int(getattr(ui_cfg, "oracle_update_interval", 10))
        except Exception:
            pass
        self.overlay = OverlayFlags()
        self.attention_channel = "sum"
        self.attention_stride = int(max(1, getattr(ui_cfg, "attention_stride", 4)))
        self._attention_cache = {"step": -999, "channel": None, "map": None}
        self.seed = int(ui_cfg.seed)
        self.fps = int(ui_cfg.fps)
        self.agent_idx = 0
        self.last_infos: dict | None = None
        self.last_obs: dict | None = None
        self.last_actions: dict | None = None
        # Траектории ограничиваются по длине, чтобы не раздувать память.
        self.trails = [[] for _ in range(self.env.n_agents)]

        self.scene_root = resolve_repo_path("scenarios") if scene_root is None else scene_root
        self.user_scene_root = resolve_repo_path("scenarios/user") if user_scene_root is None else user_scene_root
        self.scenes = load_scenes_from_roots([self.scene_root, self.user_scene_root])
        self.scene_names = [name for name, _ in self.scenes]
        self.selected_scene = ""
        self.custom_scene = None

        self.available = [
            "random",
            "zero",
            "greedy",
            "pf",
            "wall",
            "brake",
            "baseline:astar_grid",
            "baseline:flow_field",
            "baseline:greedy_safe",
            "baseline:mpc_lite",
            "baseline:separation_steering",
        ]
        if PPO_AVAILABLE:
            self.available.insert(0, "ppo")
        self.model_path = ""
        self.deterministic = True
        default_policy = "random" if "random" in self.available else self.available[0]
        if default_policy == "ppo":
            default_policy = next((p for p in self.available if p != "ppo"), "random")
        self.policy_name, self.policy = create_policy(
            default_policy,
            self.env,
            None,
            deterministic=True,
        )
        self._policy_pool = None
        self._build_policy_pool()

        self.reset_env(new_map=False)

    def reset_env(self, new_map: bool = False, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = int(seed)
        if self.custom_scene is not None:
            scene = dict(self.custom_scene)
            if new_map:
                scene["seed"] = int(scene.get("seed", 0)) + 1
            out = self.env.reset(seed=self.seed, options={"scene": scene})
        elif self.selected_scene:
            scene = dict(self.scenes[self.scene_names.index(self.selected_scene)][1])
            if new_map:
                # Сдвиг зерна заставляет пересгенерировать случайные элементы сцены.
                scene["seed"] = int(scene.get("seed", 0)) + 1
            out = self.env.reset(seed=self.seed, options={"scene": scene})
        else:
            out = self.env.reset(seed=self.seed)
        for t in self.trails:
            t.clear()
        if isinstance(out, tuple) and len(out) == 2:
            self.last_obs, self.last_infos = out
        else:
            self.last_obs, self.last_infos = out, None
        reset_policy_context(self.env, self.policy, self.seed, policy_name=self.policy_name)
        if self._policy_pool is not None:
            self._policy_pool.reset(self.seed)
        self.last_actions = None

    def step_env(self) -> bool:
        if self._policy_pool is not None:
            actions = self._policy_pool.compute(self.env, self.last_obs, self.last_infos)
        else:
            actions = _policy_actions(self.env, self.policy, self.last_obs, self.last_infos)
        self.last_actions = actions
        obs, _rewards, terms, truncs, infos = self.env.step(actions)
        self.last_obs = obs
        self.last_infos = infos
        state = self.env.get_public_state(include_oracle=False)
        for i in range(self.env.n_agents):
            p = state.pos[i]
            self.trails[i].append((float(p[0]), float(p[1])))
            if len(self.trails[i]) > 2000:
                self.trails[i] = self.trails[i][-2000:]
        done = bool(any(terms.values()) or any(truncs.values()))
        return done

    def get_state(self) -> ControllerState:
        return ControllerState(
            policy=self.policy_name,
            scene=self.selected_scene,
            seed=self.seed,
            n_agents=int(self.env.n_agents),
            model_path=self.model_path,
            deterministic=self.deterministic,
            overlay={
                "show_grid": self.overlay.show_grid,
                "show_trails": self.overlay.show_trails,
                "show_threats": self.overlay.show_threats,
                "show_attention": bool(getattr(self.overlay, "show_attention", False)),
                "attention_channel": self.attention_channel,
            },
            tunables=self.get_tunables(),
            oracle={
                "enabled": bool(getattr(self.env, "oracle_enabled", False)),
                "async": bool(getattr(self.env, "oracle_async", False)),
                "update_interval": int(getattr(self.env.config, "oracle_update_interval", 1)),
            },
        )

    def set_oracle(
        self, *, enabled: bool | None = None, async_mode: bool | None = None, interval: int | None = None
    ) -> None:
        self.env.set_oracle_options(enabled=enabled, async_mode=async_mode, update_interval=interval)

    def get_telemetry(self) -> TelemetryPayload:
        state = self.env.get_public_state(include_oracle=False)
        agents = list(self.env.possible_agents)
        oracle_path = self.env.get_oracle_path()
        wind_vec = None
        try:
            if agents:
                agent_idx = int(np.clip(self.agent_idx, 0, len(agents) - 1))
                sample = self.env.sample_wind(agent_idx)
                if np.asarray(sample).shape == (2,):
                    wind_vec = [float(sample[0]), float(sample[1])]
        except Exception:
            wind_vec = None
        payload = build_payload(
            state,
            self.last_infos,
            screen_size=self.ui_cfg.screen_size,
            agents=agents,
            agent_idx=self.agent_idx,
            last_actions=self.last_actions,
            field_size=float(self.env.config.field_size),
            goal_radius=float(self.env.goal_radius),
            grid_res=float(getattr(self.env.config, "grid_res", 1.0)),
            oracle_path=[list(p) for p in (oracle_path or [])],
            obs_provider=lambda idx: self.env.get_agent_observation(idx),
            wind=wind_vec,
        )
        return payload

    def get_telemetry_dict(self) -> dict:
        payload = self.get_telemetry()
        payload_dict = payload.to_dict()
        if getattr(self.overlay, "show_attention", False):
            attention = self._compute_attention(payload_dict)
            if attention is not None:
                payload_dict["agent_attention"] = attention
        return payload_dict

    def _compute_attention(self, payload: dict) -> list[list[float]] | None:
        if payload.get("agent_grid") is None or payload.get("agent_obs") is None:
            return None
        step = int((payload.get("stats") or {}).get("step", -1))
        channel = str(self.attention_channel or "sum").lower()
        if (
            self._attention_cache.get("map") is not None
            and self._attention_cache.get("channel") == channel
            and (step - int(self._attention_cache.get("step", -999))) < self.attention_stride
        ):
            return self._attention_cache.get("map")
        try:
            from common.runtime.xai import get_saliency_map
        except Exception:
            return None
        model = getattr(self.policy, "model", None)
        if model is None:
            return None
        obs_info = payload.get("agent_obs") or {}
        vector_dim = OBS_VECTOR_DIM
        try:
            obs_space = getattr(model, "observation_space", None)
            if obs_space is not None and hasattr(obs_space, "spaces") and "vector" in obs_space.spaces:
                vector_dim = int(np.prod(obs_space.spaces["vector"].shape))
        except Exception:
            vector_dim = OBS_VECTOR_DIM
        vec = fields_to_obs_vector(obs_info, vector_dim=vector_dim)
        if vec.size != vector_dim:
            return None
        try:
            grid = np.asarray(payload.get("agent_grid"), dtype=np.float32)
        except Exception:
            return None
        if grid.ndim != 2:
            return None
        try:
            obs_space = getattr(model, "observation_space", None)
            if obs_space is not None and hasattr(obs_space, "spaces") and "grid" in obs_space.spaces:
                grid_shape = tuple(int(x) for x in obs_space.spaces["grid"].shape)
                if len(grid_shape) == 3 and grid.shape == grid_shape[1:]:
                    grid = np.tile(grid[None, :, :], (grid_shape[0], 1, 1)).astype(np.float32)
        except Exception:
            pass
        try:
            action_dim = int(np.prod(model.action_space.shape))
        except Exception:
            action_dim = 2
        if channel == "x":
            dims = [0]
        elif channel == "y":
            dims = [1]
        else:
            dims = [0, 1]
        dims = [d for d in dims if d < action_dim]
        if not dims:
            dims = [0]
        maps = []
        for dim in dims:
            try:
                heat = get_saliency_map(model, {"grid": grid, "vector": vec}, action_dim=dim, normalize=False)
                maps.append(heat)
            except Exception:
                return None
        heatmap = np.sum(maps, axis=0)
        max_val = float(np.max(heatmap)) if heatmap is not None else 0.0
        if max_val > 1e-8:
            heatmap = heatmap / max_val
        out = heatmap.astype(np.float32).tolist()
        self._attention_cache["step"] = step
        self._attention_cache["channel"] = channel
        self._attention_cache["map"] = out
        return out

    def _close_policy_pool(self) -> None:
        if self._policy_pool is not None:
            self._policy_pool.shutdown()
            self._policy_pool = None

    def _build_policy_pool(self) -> None:
        self._close_policy_pool()
        self._policy_pool = build_policy_pool(self.policy, self.policy_name, self.env, self.ui_cfg, log=logger)

    def close(self) -> None:
        self._close_policy_pool()
        try:
            self.env.close()
        except Exception:
            pass

    def __del__(self) -> None:
        try:
            self._close_policy_pool()
        except Exception:
            pass

    def set_policy(self, name: str) -> None:
        self.policy_name, self.policy = create_policy(name, self.env, self.model_path or None, self.deterministic)
        self._build_policy_pool()
        reset_policy_context(self.env, self.policy, self.seed, policy_name=self.policy_name)

    def set_scene(self, name: str) -> None:
        if name and name in self.scene_names:
            self.selected_scene = name
        else:
            self.selected_scene = ""
        self.custom_scene = None
        self.reset_env(new_map=False)

    def set_custom_scene(self, scene: dict | None) -> None:
        self.custom_scene = dict(scene) if isinstance(scene, dict) else None
        self.selected_scene = ""

    def reload_scenes(self) -> None:
        self.scenes = load_scenes_from_roots([self.scene_root, self.user_scene_root])
        self.scene_names = [name for name, _ in self.scenes]

    def set_seed(self, seed: int) -> None:
        self.seed = int(seed)

    def set_model(self, model_path: str, deterministic: bool | None = None) -> None:
        self.model_path = str(model_path)
        if deterministic is not None:
            self.deterministic = bool(deterministic)
        if self.model_path:
            self.policy_name, self.policy = create_policy("ppo", self.env, self.model_path, self.deterministic)
            self._build_policy_pool()
            reset_policy_context(self.env, self.policy, self.seed, policy_name=self.policy_name)

    def set_deterministic(self, value: bool) -> None:
        self.deterministic = bool(value)
        if self.policy_name.startswith("ppo") and self.model_path:
            self.set_model(self.model_path, deterministic=self.deterministic)

    def apply_tunables(self, params: dict) -> None:
        apply_runtime_tunables(self.env, params)

    def get_tunables(self) -> dict:
        return collect_tunables(self.env)
