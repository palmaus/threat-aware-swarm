"""Управление сценами, стенами и подготовкой окружения."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from env.scenes.threats import threat_from_config


class SceneManager:
    def __init__(self, env, spawn_controller, rng: np.random.Generator | None = None) -> None:
        self._env = env
        self._spawn = spawn_controller
        self._rng = rng if rng is not None else np.random.default_rng()

    def set_rng(self, rng: np.random.Generator | None) -> None:
        if rng is None:
            return
        self._rng = rng

    @contextmanager
    def transaction(self):
        """Откатывает scene-мутации, если загрузка карты/угроз упала посередине."""

        snapshot = self._env.capture_runtime_snapshot()
        try:
            yield
        except Exception:
            self._env.restore_runtime_snapshot(snapshot)
            raise

    def apply_scene(self, scene: dict, *, apply_threats: bool = True) -> None:
        if not scene:
            return
        self.apply_scene_layout(scene)
        if apply_threats:
            self.apply_scene_threats(scene)
        self.apply_scene_agents(scene)

    def apply_scene_layout(self, scene: dict) -> None:
        if not scene:
            return
        if scene.get("field_size") is not None:
            self._env.set_episode_field_size(float(scene["field_size"]))
        if scene.get("max_steps") is not None:
            self._env.max_steps = int(scene["max_steps"])
        if scene.get("target_pos") is not None:
            target = np.asarray(scene["target_pos"], dtype=np.float32).reshape(-1)
            if target.size < 2:
                raise ValueError("scene.target_pos must contain at least two coordinates")
            self._env.target_pos = np.clip(target[:2], 0.0, self._env.config.field_size).astype(np.float32)
        self._env.configure_target_motion(scene.get("target_motion"))
        walls = list(scene.get("walls", []) or [])
        self.set_walls(walls)
        circle_obstacles = []
        for raw in scene.get("circle_obstacles", []) or []:
            try:
                c_pos = np.asarray(raw.get("pos"), dtype=np.float32)
                c_rad = float(raw.get("radius", 0.0))
            except Exception:
                continue
            circle_obstacles.append((c_pos, c_rad))
        if circle_obstacles:
            self._env.set_static_circles(circle_obstacles)
        else:
            self._env.set_static_circles([])

    def apply_scene_agents(self, scene: dict) -> None:
        if not scene:
            return
        if scene.get("agents_pos") is not None:
            source_positions = list(scene["agents_pos"])
            if not source_positions:
                return
            n_agents = int(self._env.config.n_agents)
            if len(source_positions) < n_agents:
                pos_list = [source_positions[i % len(source_positions)] for i in range(n_agents)]
            else:
                pos_list = source_positions[:n_agents]
            arr = np.asarray(pos_list, dtype=np.float32)
            if arr.shape[0] != self._env.config.n_agents:
                arr = arr[: self._env.config.n_agents]
            self._env.set_agent_positions(np.clip(arr, 0, self._env.config.field_size))
        elif scene.get("start_centers") is not None:
            centers = scene.get("start_centers") or []
            sigma = float(scene.get("start_sigma", 2.0))
            chunks = []
            n = int(self._env.config.n_agents)
            k = max(1, len(centers))
            base = n // k
            extra = n % k
            for i, center in enumerate(centers):
                count = base + (1 if i < extra else 0)
                if count <= 0:
                    continue
                c = np.asarray(center, dtype=np.float32)
                existing = np.vstack(chunks) if chunks else None
                chunks.append(
                    self._spawn.sample_safe_cluster(c, n=count, sigma=sigma, existing_positions=existing)
                )
            if chunks:
                self._env.set_agent_positions(np.vstack(chunks))
        elif scene.get("start_center") is not None:
            center = np.asarray(scene.get("start_center"), dtype=np.float32)
            sigma = float(scene.get("start_sigma", 2.0))
            self._env.set_agent_positions(self._spawn.sample_safe_cluster(center, sigma=sigma))

    def apply_scene_map(self, scene: dict) -> None:
        self.apply_scene_layout(scene)

    def apply_scene_threats(self, scene: dict) -> None:
        if not scene:
            return
        # Угрозы пересоздаются при загрузке сцены.
        self._env.clear_threats()

        def _add_threats(items, forced_type: str | None = None):
            for raw in items or []:
                if not isinstance(raw, dict):
                    continue
                cfg = dict(raw)
                if forced_type and "type" not in cfg and "behavior" not in cfg:
                    cfg["type"] = forced_type
                threat = threat_from_config(cfg, rng=self._rng)
                self._env.add_threat_object(threat)

        _add_threats(scene.get("threats", []) or [])
        _add_threats(scene.get("static_threats", []) or [], forced_type="static")
        _add_threats(scene.get("dynamic_threats", []) or [], forced_type="linear")

    def set_walls(self, walls) -> None:
        self._env.set_walls(walls)

    def generate_random_walls(self) -> list[dict[str, float]]:
        walls = []
        count = max(0, int(self._env.config.wall_count))
        if count <= 0:
            return walls
        size = float(self._env.config.field_size)
        wmin = float(self._env.config.wall_size_min)
        wmax = float(self._env.config.wall_size_max)
        for _ in range(count):
            w = self._rng.uniform(wmin, wmax)
            h = self._rng.uniform(wmin, wmax)
            x1 = self._rng.uniform(0.0, max(0.0, size - w))
            y1 = self._rng.uniform(0.0, max(0.0, size - h))
            walls.append({"x1": float(x1), "y1": float(y1), "x2": float(x1 + w), "y2": float(y1 + h)})
        return walls
