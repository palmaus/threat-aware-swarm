import numpy as np

from common.physics.model import apply_accel_dynamics_vel, compute_drag_factor
from common.physics.walls import circle_hits_any as _common_circle_hits_any
from common.physics.walls import circle_hits_any_batch as _common_circle_hits_any_batch
from common.physics.walls import circle_rect_normal as _common_circle_rect_normal
from common.physics.walls import circle_rect_normals_batch as _common_circle_rect_normals_batch
from common.physics.walls import resolve_wall_slide as _common_resolve_wall_slide
from common.physics.walls import resolve_wall_slide_batch as _common_resolve_wall_slide_batch
from common.physics.walls import resolve_wall_slide_batch_inner as _common_resolve_wall_slide_batch_inner
from env.config import EnvConfig
from env.physics.wind import WindField
from env.scenes.threats import BaseThreat, StaticThreat, is_dynamic_threat

AGENT_ACTIVE = np.int8(0)
AGENT_CRASHING = np.int8(1)
AGENT_DEAD = np.int8(2)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    out = np.zeros_like(vectors, dtype=np.float32)
    mask = norms[:, 0] > 1e-6
    if np.any(mask):
        out[mask] = vectors[mask] / norms[mask]
    return out


class PhysicsCore:
    def __init__(self, config: EnvConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.n = config.n_agents
        self.rng = rng if rng is not None else np.random.default_rng()

        # Базовые состояния агента и среды.
        self.agents_pos = np.zeros((self.n, 2), dtype=np.float32)
        self.agents_vel = np.zeros(
            (self.n, 2), dtype=np.float32
        )  # Скорость нужна в наблюдениях, поэтому хранится отдельно.
        self.agents_active = np.ones(self.n, dtype=bool)
        self.agent_state = np.full(self.n, AGENT_ACTIVE, dtype=np.int8)
        cap = float(getattr(self.config.battery, "capacity", 0.0))
        self.energy = np.full(self.n, cap, dtype=np.float32)

        self.threats: list[BaseThreat] = []
        self.static_threats: list[BaseThreat] = []
        self.dynamic_threats: list[BaseThreat] = []
        self.threat_pos = np.zeros((0, 2), dtype=np.float32)
        self.threat_vel = np.zeros((0, 2), dtype=np.float32)
        self.threat_radius = np.zeros((0,), dtype=np.float32)
        self.threat_intensity = np.zeros((0,), dtype=np.float32)
        self.threat_oracle_block = np.zeros((0,), dtype=bool)
        self.threat_is_dynamic = np.zeros((0,), dtype=bool)
        self.walls = []
        self.circle_obstacles_pos = np.zeros((0, 2), dtype=np.float32)
        self.circle_obstacles_radius = np.zeros((0,), dtype=np.float32)
        self.last_collision = np.zeros(self.n, dtype=bool)
        self.last_collision_speed = np.zeros(self.n, dtype=np.float32)
        self.time_step = 0
        self.threat_speed_scale = 1.0
        self._wind_field: WindField | None = None
        self._thrust_buf = np.zeros((self.n, 2), dtype=np.float32)
        self._thr_norm_buf = np.zeros((self.n,), dtype=np.float32)
        self._thr_scale_buf = np.ones((self.n,), dtype=np.float32)
        self._wind_buf = np.zeros((self.n, 2), dtype=np.float32)
        self._accel_buf = np.zeros((self.n, 2), dtype=np.float32)
        self._next_vel_buf = np.zeros((self.n, 2), dtype=np.float32)
        self._next_pos_buf = np.zeros((self.n, 2), dtype=np.float32)

    def add_threat(self, position, radius, intensity, *, oracle_block: bool = True):
        threat = StaticThreat(
            position=np.array(position, dtype=np.float32),
            radius=radius,
            intensity=intensity,
            oracle_block=bool(oracle_block),
            rng=self.rng,
        )
        self.add_threat_obj(threat)

    def add_threat_obj(self, threat: BaseThreat):
        self.threats.append(threat)
        if is_dynamic_threat(threat):
            self.dynamic_threats.append(threat)
        else:
            self.static_threats.append(threat)
        self._sync_threat_arrays()

    def reset(self):
        # Сбрасываем только состояние симулятора; расстановкой агентов управляет среда.
        self.time_step = 0
        self.agent_state[:] = AGENT_ACTIVE
        self.agents_active[:] = True
        cap = float(getattr(self.config.battery, "capacity", 0.0))
        self.energy[:] = cap
        self.threats = []
        self.static_threats = []
        self.dynamic_threats = []
        self._sync_threat_arrays()
        self.walls = []
        self.circle_obstacles_pos = np.zeros((0, 2), dtype=np.float32)
        self.circle_obstacles_radius = np.zeros((0,), dtype=np.float32)
        self.last_collision[:] = False
        self.last_collision_speed[:] = 0.0
        # Базовая инициализация на случай вызова напрямую.
        self.agents_pos = self.rng.uniform(0, 20, (self.n, 2))
        self.agents_vel = np.zeros((self.n, 2), dtype=np.float32)
        if self._wind_field is not None:
            self._wind_field.reset()
        return self.get_state()

    def set_rng(self, rng: np.random.Generator | None) -> None:
        if rng is None:
            return
        self.rng = rng

    def set_walls(self, walls: list[tuple[float, float, float, float]]):
        self.walls = list(walls or [])
        self._sync_wall_sdf()

    def set_circle_obstacles(self, circles: list[tuple[np.ndarray, float]] | None) -> None:
        if not circles:
            self.circle_obstacles_pos = np.zeros((0, 2), dtype=np.float32)
            self.circle_obstacles_radius = np.zeros((0,), dtype=np.float32)
            return
        pos = []
        rad = []
        for item in circles:
            try:
                c_pos, c_rad = item
            except Exception:
                continue
            pos.append(np.asarray(c_pos, dtype=np.float32))
            rad.append(float(c_rad))
        if pos:
            self.circle_obstacles_pos = np.asarray(pos, dtype=np.float32)
            self.circle_obstacles_radius = np.asarray(rad, dtype=np.float32)
        else:
            self.circle_obstacles_pos = np.zeros((0, 2), dtype=np.float32)
            self.circle_obstacles_radius = np.zeros((0,), dtype=np.float32)

    def set_wind_field(self, field: WindField | None) -> None:
        self._wind_field = field

    def clone(self):
        cloned = PhysicsCore(self.config)
        cloned.n = self.n
        cloned.agents_pos = self.agents_pos.copy()
        cloned.agents_vel = self.agents_vel.copy()
        cloned.agents_active = self.agents_active.copy()
        cloned.agent_state = self.agent_state.copy()
        cloned.energy = self.energy.copy()
        cloned.threats = [t.copy() for t in self.threats]
        cloned.static_threats = [t for t in cloned.threats if not is_dynamic_threat(t)]
        cloned.dynamic_threats = [t for t in cloned.threats if is_dynamic_threat(t)]
        cloned._sync_threat_arrays()
        cloned.walls = list(self.walls)
        cloned.circle_obstacles_pos = self.circle_obstacles_pos.copy()
        cloned.circle_obstacles_radius = self.circle_obstacles_radius.copy()
        cloned.time_step = int(self.time_step)
        cloned.threat_speed_scale = float(self.threat_speed_scale)
        cloned._sync_wall_sdf()
        return cloned

    def _ensure_step_buffers(self) -> None:
        if self._thrust_buf.shape == self.agents_pos.shape:
            return
        shape = self.agents_pos.shape
        n = shape[0]
        self._thrust_buf = np.zeros(shape, dtype=np.float32)
        self._thr_norm_buf = np.zeros((n,), dtype=np.float32)
        self._thr_scale_buf = np.ones((n,), dtype=np.float32)
        self._wind_buf = np.zeros(shape, dtype=np.float32)
        self._accel_buf = np.zeros(shape, dtype=np.float32)
        self._next_vel_buf = np.zeros(shape, dtype=np.float32)
        self._next_pos_buf = np.zeros(shape, dtype=np.float32)

    def step(self, action_vectors: np.ndarray):
        """
        action_vectors: (N, 2) — векторы тяги (Н).
        """
        if self.n == 0:
            return self.get_state()
        self._ensure_step_buffers()

        dt = float(self.config.dt)
        mass = float(getattr(self.config.physics, "mass", 1.0))
        max_thrust = float(getattr(self.config.physics, "max_thrust", 0.0))
        drag_coeff = float(getattr(self.config.physics, "drag_coeff", 0.0))

        thrust = self._thrust_buf
        np.copyto(thrust, np.asarray(action_vectors, dtype=np.float32))
        thr_norm_scaled = self._thr_norm_buf
        np.sqrt(np.sum(thrust * thrust, axis=1, dtype=np.float32), out=thr_norm_scaled)
        thr_scale = self._thr_scale_buf
        thr_scale.fill(1.0)
        if max_thrust > 0.0:
            over = thr_norm_scaled > max_thrust
            if np.any(over):
                thr_scale[over] = max_thrust / (thr_norm_scaled[over] + 1e-8)
        thrust *= thr_scale[:, None]
        thr_norm_scaled *= thr_scale

        active_mask = self.agent_state == AGENT_ACTIVE
        thrust[~active_mask] = 0.0

        cap = float(getattr(self.config.battery, "capacity", 0.0))
        drain_hover = float(getattr(self.config.battery, "drain_hover", 0.0))
        drain_thrust = float(getattr(self.config.battery, "drain_thrust", 0.0))
        if cap > 0.0:
            thr_norm_scaled[~active_mask] = 0.0
            drain = (drain_hover + (thr_norm_scaled * drain_thrust)) * float(dt)
            self.energy = np.maximum(0.0, self.energy - drain.astype(np.float32))
            depleted = self.energy <= 0.0
            if np.any(depleted & (self.agent_state == AGENT_ACTIVE)):
                self.agent_state[depleted] = AGENT_CRASHING
                active_mask = self.agent_state == AGENT_ACTIVE
                thrust[~active_mask] = 0.0

        wind = self._wind_buf
        wind.fill(0.0)
        if self._wind_field is not None:
            self._wind_field.step(dt)
            np.copyto(wind, np.asarray(self._wind_field.sample(self.agents_pos), dtype=np.float32))

        accel = self._accel_buf
        np.multiply(thrust, np.float32(1.0 / max(mass, 1e-6)), out=accel)
        dyn_mask = self.agent_state != AGENT_DEAD
        max_speed = float(self.config.max_speed)
        drag = compute_drag_factor(drag_coeff, mass, dt)
        next_vel = apply_accel_dynamics_vel(
            self.agents_vel,
            accel,
            dt,
            drag=drag,
            max_speed=max_speed,
            wind=wind,
            mask=dyn_mask,
            out=self._next_vel_buf,
        )

        next_pos = self._next_pos_buf
        np.copyto(next_pos, self.agents_pos)
        self.last_collision[:] = False
        self.last_collision_speed[:] = 0.0
        dyn_indices = np.where(dyn_mask)[0]
        if dyn_indices.size:
            if self.walls:
                radius = float(max(0.0, self.config.agent_radius))
                friction = float(np.clip(self.config.wall_friction, 0.0, 1.0))
                pos_dyn = self.agents_pos[dyn_indices]
                vel_dyn = next_vel[dyn_indices]
                if getattr(self, "_wall_sdf", None) is not None:
                    new_pos, new_vel, impacts = _resolve_wall_slide_sdf_batch(
                        pos_dyn,
                        vel_dyn,
                        dt,
                        self._wall_sdf,
                        self._wall_sdf_grad,
                        float(self._wall_sdf_res),
                        radius,
                        friction,
                    )
                else:
                    new_pos, new_vel, impacts = _resolve_wall_slide_batch(
                        pos_dyn,
                        vel_dyn,
                        dt,
                        self.walls,
                        radius,
                        friction,
                    )
                next_vel[dyn_indices] = new_vel
                next_pos[dyn_indices] = new_pos
                hit_mask = impacts > 0.0
                if np.any(hit_mask):
                    hit_idx = dyn_indices[hit_mask]
                    self.last_collision[hit_idx] = True
                    self.last_collision_speed[hit_idx] = impacts[hit_mask]
            else:
                next_pos[dyn_indices] += next_vel[dyn_indices] * dt

            circles_pos, circles_rad = _collect_circle_obstacles(
                self.circle_obstacles_pos,
                self.circle_obstacles_radius,
                self.agents_pos,
                self.agent_state,
                float(self.config.agent_radius),
            )
            if circles_pos.size:
                radius = float(max(0.0, self.config.agent_radius))
                friction = float(np.clip(self.config.wall_friction, 0.0, 1.0))
                pos_dyn = next_pos[dyn_indices]
                vel_dyn = next_vel[dyn_indices]
                new_pos, new_vel, impacts = _resolve_circle_obstacles_batch(
                    pos_dyn,
                    vel_dyn,
                    dt,
                    circles_pos,
                    circles_rad,
                    radius,
                    friction,
                )
                next_vel[dyn_indices] = new_vel
                next_pos[dyn_indices] = new_pos
                hit_mask = impacts > 0.0
                if np.any(hit_mask):
                    hit_idx = dyn_indices[hit_mask]
                    self.last_collision[hit_idx] = True
                    self.last_collision_speed[hit_idx] = np.maximum(
                        self.last_collision_speed[hit_idx], impacts[hit_mask]
                    )

        np.copyto(self.agents_vel, next_vel)
        self.agents_pos[dyn_indices] = next_pos[dyn_indices]

        crashing = self.agent_state == AGENT_CRASHING
        if np.any(crashing):
            speeds = np.linalg.norm(self.agents_vel[crashing], axis=1)
            stopped = speeds < 0.05
            if np.any(stopped):
                crash_idx = np.where(crashing)[0][stopped]
                self.agent_state[crash_idx] = AGENT_DEAD
                self.agents_vel[crash_idx] = 0.0

        self.agents_active = self.agent_state == AGENT_ACTIVE

        # Этап 3: границы поля и обновление угроз.
        self._apply_boundaries()
        self._update_threats()
        self._update_survival()

        self.time_step += 1
        return self.get_state()

    def _apply_boundaries(self):
        # Жесткое ограничение поля: агент не выходит за [0, field_size].
        # Штрафы за приближение к стенам начисляются на уровне награды.
        np.clip(self.agents_pos, 0, self.config.field_size, out=self.agents_pos)

    def _update_threats(self):
        if not self.dynamic_threats:
            return
        if not self.walls:
            self._update_dynamic_threats_batch_no_walls()
            self._sync_threat_arrays()
            return
        for t in self.dynamic_threats:
            try:
                t.update(
                    float(self.config.dt) * float(self.threat_speed_scale),
                    float(self.config.field_size),
                    self.walls,
                    self.agents_pos,
                    self.agent_state != AGENT_DEAD,
                )
            except Exception:
                continue
        self._sync_threat_arrays()

    def _update_dynamic_threats_batch_no_walls(self) -> None:
        if not self.dynamic_threats:
            return
        dt = float(self.config.dt) * float(self.threat_speed_scale)
        if dt <= 0.0:
            return
        field_size = float(self.config.field_size)
        active_agents = self.agent_state != AGENT_DEAD
        agent_pos = self.agents_pos

        linear_idx = [i for i, t in enumerate(self.dynamic_threats) if getattr(t, "kind", "") == "linear"]
        brownian_idx = [i for i, t in enumerate(self.dynamic_threats) if getattr(t, "kind", "") == "brownian"]
        other_idx = [i for i, t in enumerate(self.dynamic_threats) if i not in linear_idx and i not in brownian_idx]

        if linear_idx:
            threats = [self.dynamic_threats[i] for i in linear_idx]
            for threat in threats:
                updater = getattr(threat, "_update_radius", None)
                if updater is not None:
                    updater(dt)
            pos = np.asarray([t.position for t in threats], dtype=np.float32)
            vel = np.asarray([t.velocity for t in threats], dtype=np.float32)
            rad = np.asarray([float(t.radius) for t in threats], dtype=np.float32)
            pos += vel * dt
            self._apply_threat_boundary_bounce(pos, vel, rad, field_size)
            for offset, threat in enumerate(threats):
                threat.position = pos[offset].astype(np.float32)
                threat.velocity = vel[offset].astype(np.float32)

        if brownian_idx:
            threats = [self.dynamic_threats[i] for i in brownian_idx]
            for threat in threats:
                updater = getattr(threat, "_update_radius", None)
                if updater is not None:
                    updater(dt)
            pos = np.asarray([t.position for t in threats], dtype=np.float32)
            vel = np.asarray([t.velocity for t in threats], dtype=np.float32)
            speed = np.asarray([float(getattr(t, "speed", 0.0)) for t in threats], dtype=np.float32)
            rad = np.asarray([float(t.radius) for t in threats], dtype=np.float32)
            jitter = np.asarray(
                [t.rng.normal(0.0, float(getattr(t, "noise_scale", 0.0)), size=2).astype(np.float32) for t in threats],
                dtype=np.float32,
            )
            direction = _normalize_rows(vel + jitter)
            zero_mask = np.linalg.norm(direction, axis=1) <= 1e-6
            if np.any(zero_mask):
                fallback = np.asarray(
                    [threats[i].rng.normal(size=2).astype(np.float32) for i in np.where(zero_mask)[0]],
                    dtype=np.float32,
                )
                direction[zero_mask] = _normalize_rows(fallback)
            vel = direction * speed[:, None]
            pos += vel * dt
            self._apply_threat_boundary_bounce(pos, vel, rad, field_size)
            for offset, threat in enumerate(threats):
                threat.position = pos[offset].astype(np.float32)
                threat.velocity = vel[offset].astype(np.float32)

        for idx in other_idx:
            threat = self.dynamic_threats[idx]
            try:
                threat.update(dt, field_size, None, agent_pos, active_agents)
            except Exception:
                continue

    def _apply_threat_boundary_bounce(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        radius: np.ndarray,
        field_size: float,
    ) -> None:
        min_pos = np.maximum(radius, 0.0)
        max_pos = np.maximum(field_size - radius, min_pos)
        for axis in (0, 1):
            low = pos[:, axis] < min_pos
            if np.any(low):
                pos[low, axis] = min_pos[low]
                vel[low, axis] = np.abs(vel[low, axis])
            high = pos[:, axis] > max_pos
            if np.any(high):
                pos[high, axis] = max_pos[high]
                vel[high, axis] = -np.abs(vel[high, axis])

    def _update_survival(self):
        # Вероятностная модель смерти по интенсивности угроз.
        active_indices = np.where(self.agent_state == AGENT_ACTIVE)[0]
        if len(active_indices) == 0:
            return
        if not self.threats:
            return

        positions = self.agents_pos[active_indices]
        t_pos = self.threat_pos
        t_rad = self.threat_radius
        t_int = self.threat_intensity
        if t_pos.size == 0:
            return
        diff = positions[:, None, :] - t_pos[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        inside = dist2 <= (t_rad[None, :] * t_rad[None, :])
        survival_probs = np.prod(1.0 - (inside * t_int[None, :]), axis=1)
        random_vals = self.rng.random(len(active_indices))
        survived = random_vals < survival_probs

        killed_indices = active_indices[~survived]
        if killed_indices.size:
            self.agent_state[killed_indices] = AGENT_CRASHING
            self.agents_active[killed_indices] = False
            self.energy[killed_indices] = 0.0

    def _sync_threat_arrays(self) -> None:
        if not self.threats:
            self.threat_pos = np.zeros((0, 2), dtype=np.float32)
            self.threat_vel = np.zeros((0, 2), dtype=np.float32)
            self.threat_radius = np.zeros((0,), dtype=np.float32)
            self.threat_intensity = np.zeros((0,), dtype=np.float32)
            self.threat_oracle_block = np.zeros((0,), dtype=bool)
            self.threat_is_dynamic = np.zeros((0,), dtype=bool)
            return
        t_pos = []
        t_vel = []
        t_rad = []
        t_int = []
        t_block = []
        t_dyn = []
        for t in self.threats:
            t_pos.append(np.asarray(getattr(t, "position", [0.0, 0.0]), dtype=np.float32))
            t_vel.append(np.asarray(getattr(t, "velocity", getattr(t, "vel", [0.0, 0.0])), dtype=np.float32))
            t_rad.append(float(getattr(t, "radius", 0.0)))
            t_int.append(float(getattr(t, "intensity", 0.0)))
            t_block.append(bool(getattr(t, "oracle_block", False)))
            t_dyn.append(bool(is_dynamic_threat(t)))
        self.threat_pos = np.asarray(t_pos, dtype=np.float32)
        self.threat_vel = np.asarray(t_vel, dtype=np.float32)
        self.threat_radius = np.asarray(t_rad, dtype=np.float32)
        self.threat_intensity = np.asarray(t_int, dtype=np.float32)
        self.threat_oracle_block = np.asarray(t_block, dtype=bool)
        self.threat_is_dynamic = np.asarray(t_dyn, dtype=bool)

    def _sync_wall_sdf(self) -> None:
        self._wall_sdf = None
        self._wall_sdf_grad = None
        self._wall_sdf_res = None
        if not self.walls:
            return
        try:
            from scipy import ndimage

            use_scipy = True
        except Exception:
            use_scipy = False
        res = float(getattr(self.config, "grid_res", 1.0))
        if res <= 0.0:
            return
        size = float(self.config.field_size)
        width = int(np.ceil(size / res)) + 1
        if use_scipy:
            grid = np.zeros((width, width), dtype=bool)
            for x1, y1, x2, y2 in self.walls:
                ix1 = max(0, int(np.floor(x1 / res)))
                iy1 = max(0, int(np.floor(y1 / res)))
                ix2 = min(width - 1, int(np.ceil(x2 / res)))
                iy2 = min(width - 1, int(np.ceil(y2 / res)))
                grid[iy1 : iy2 + 1, ix1 : ix2 + 1] = True
            outside = ndimage.distance_transform_edt(~grid) * res
            inside = ndimage.distance_transform_edt(grid) * res
            sdf = outside - inside
        else:
            xs = (np.arange(width, dtype=np.float32) * res) + (res / 2.0)
            ys = (np.arange(width, dtype=np.float32) * res) + (res / 2.0)
            grid_x, grid_y = np.meshgrid(xs, ys)
            outside_dist = np.full((width, width), np.inf, dtype=np.float32)
            inside_dist = np.full((width, width), np.inf, dtype=np.float32)
            for x1, y1, x2, y2 in self.walls:
                x1 = float(x1)
                y1 = float(y1)
                x2 = float(x2)
                y2 = float(y2)
                dx = np.maximum(np.maximum(x1 - grid_x, 0.0), grid_x - x2)
                dy = np.maximum(np.maximum(y1 - grid_y, 0.0), grid_y - y2)
                dist_out = np.sqrt(dx * dx + dy * dy)
                outside_dist = np.minimum(outside_dist, dist_out)
                inside_mask = (grid_x >= x1) & (grid_x <= x2) & (grid_y >= y1) & (grid_y <= y2)
                if np.any(inside_mask):
                    dist_edge = np.minimum.reduce([grid_x - x1, x2 - grid_x, grid_y - y1, y2 - grid_y])
                    inside_dist[inside_mask] = np.minimum(inside_dist[inside_mask], dist_edge[inside_mask])
            sdf = outside_dist
            inside_mask = np.isfinite(inside_dist)
            sdf[inside_mask] = -inside_dist[inside_mask]
        grad_y, grad_x = np.gradient(sdf)
        norm = np.sqrt(grad_x * grad_x + grad_y * grad_y) + 1e-6
        grad_x = grad_x / norm
        grad_y = grad_y / norm
        self._wall_sdf = sdf.astype(np.float32)
        self._wall_sdf_grad = np.stack([grad_x, grad_y], axis=-1).astype(np.float32)
        self._wall_sdf_res = res

    def get_wall_distances(self, agent_idx):
        """
        Возвращает нормализованные расстояния до 4-х стен [Left, Right, Down, Up].
        Диапазон [0, 1].
        """
        pos = self.agents_pos[agent_idx]
        size = self.config.field_size

        d_left = pos[0] / size
        d_right = (size - pos[0]) / size
        d_down = pos[1] / size
        d_up = (size - pos[1]) / size

        return np.array([d_left, d_right, d_down, d_up], dtype=np.float32)

    def get_wall_distances_batch(self) -> np.ndarray:
        """Батч-версия get_wall_distances для всех агентов."""
        pos = self.agents_pos
        size = float(self.config.field_size)
        d_left = pos[:, 0] / size
        d_right = (size - pos[:, 0]) / size
        d_down = pos[:, 1] / size
        d_up = (size - pos[:, 1]) / size
        return np.stack([d_left, d_right, d_down, d_up], axis=1).astype(np.float32)
    def get_state(self):
        return self.agents_pos.copy(), self.agents_active.copy()


def _point_in_walls(pos: np.ndarray, walls: list[tuple[float, float, float, float]]) -> bool:
    x = float(pos[0])
    y = float(pos[1])
    for x1, y1, x2, y2 in walls:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def _resolve_wall_slide(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    walls: list[tuple[float, float, float, float]],
    radius: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    return _common_resolve_wall_slide(pos, vel, dt, walls, radius, friction)


def _resolve_wall_slide_batch(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    walls: list[tuple[float, float, float, float]],
    radius: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _common_resolve_wall_slide_batch(pos, vel, dt, walls, radius, friction)


def _resolve_wall_slide_sdf_batch(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    sdf: np.ndarray,
    sdf_grad: np.ndarray,
    res: float,
    radius: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    n = pos.shape[0]
    impacts = np.zeros(n, dtype=np.float32)
    if n == 0:
        return pos, vel, impacts
    if sdf is None or sdf.size == 0:
        return pos + (vel * dt), vel, impacts

    height, width = sdf.shape
    next_pos = pos + (vel * dt)
    gx = np.clip(np.floor(next_pos[:, 0] / res), 0, width - 1).astype(np.int32)
    gy = np.clip(np.floor(next_pos[:, 1] / res), 0, height - 1).astype(np.int32)
    dist = sdf[gy, gx]
    hit_mask = dist < float(radius)
    if not np.any(hit_mask):
        return next_pos, vel, impacts

    normals = sdf_grad[gy, gx]
    v = vel.copy()
    ncomp = np.sum(v * normals, axis=1)
    hit_dir = hit_mask & (ncomp < 0.0)
    if np.any(hit_dir):
        v[hit_dir] = v[hit_dir] - (normals[hit_dir] * ncomp[hit_dir][:, None])
        if friction > 0.0:
            v[hit_dir] = v[hit_dir] * (1.0 - friction)
        impacts[hit_dir] = np.maximum(impacts[hit_dir], (-ncomp[hit_dir]).astype(np.float32))

    next_pos = pos + (v * dt)
    gx = np.clip(np.floor(next_pos[:, 0] / res), 0, width - 1).astype(np.int32)
    gy = np.clip(np.floor(next_pos[:, 1] / res), 0, height - 1).astype(np.int32)
    dist2 = sdf[gy, gx]
    blocked = dist2 < float(radius)
    if np.any(blocked):
        next_pos[blocked] = pos[blocked]
        v[blocked] = 0.0
    return next_pos, v, impacts


def _resolve_wall_slide_batch_inner(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    walls: list[tuple[float, float, float, float]],
    radius: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _common_resolve_wall_slide_batch_inner(pos, vel, dt, walls, radius, friction)


def _circle_hits_any_batch(
    centers: np.ndarray,
    walls: list[tuple[float, float, float, float]],
    radius: float,
) -> np.ndarray:
    return _common_circle_hits_any_batch(centers, walls, radius)


def _collect_circle_obstacles(
    static_pos: np.ndarray,
    static_rad: np.ndarray,
    agents_pos: np.ndarray,
    agent_state: np.ndarray,
    agent_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    dead_mask = agent_state == AGENT_DEAD
    dead_pos = agents_pos[dead_mask]
    if dead_pos.size == 0 and static_pos.size == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    parts_pos = []
    parts_rad = []
    if static_pos.size:
        parts_pos.append(np.asarray(static_pos, dtype=np.float32))
        parts_rad.append(np.asarray(static_rad, dtype=np.float32))
    if dead_pos.size:
        parts_pos.append(np.asarray(dead_pos, dtype=np.float32))
        parts_rad.append(np.full((dead_pos.shape[0],), float(agent_radius), dtype=np.float32))
    pos = np.vstack(parts_pos) if parts_pos else np.zeros((0, 2), dtype=np.float32)
    rad = np.concatenate(parts_rad) if parts_rad else np.zeros((0,), dtype=np.float32)
    return pos, rad


def _resolve_circle_obstacles_batch(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    circles_pos: np.ndarray,
    circles_rad: np.ndarray,
    agent_radius: float,
    friction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32)
    vel = np.asarray(vel, dtype=np.float32)
    n = pos.shape[0]
    impacts = np.zeros(n, dtype=np.float32)
    if n == 0 or circles_pos.size == 0:
        return pos + (vel * dt), vel, impacts

    next_pos = pos + (vel * dt)
    for i in range(n):
        p = next_pos[i]
        v = vel[i]
        hit_speed = 0.0
        for j in range(circles_pos.shape[0]):
            c = circles_pos[j]
            r = float(circles_rad[j]) + float(agent_radius)
            diff = p - c
            dist2 = float(diff[0] * diff[0] + diff[1] * diff[1])
            if dist2 >= (r * r):
                continue
            dist = float(np.sqrt(max(dist2, 1e-8)))
            normal = diff / max(dist, 1e-6)
            ncomp = float(np.dot(v, normal))
            if ncomp < 0.0:
                v = v - (normal * ncomp)
                if friction > 0.0:
                    v = v * (1.0 - friction)
                hit_speed = max(hit_speed, -ncomp)
            p = c + normal * r
        next_pos[i] = p
        vel[i] = v
        impacts[i] = float(hit_speed)
    return next_pos, vel, impacts


def _circle_rect_normals_batch(
    centers: np.ndarray,
    rect: tuple[float, float, float, float],
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    return _common_circle_rect_normals_batch(centers, rect, radius)


def _circle_hits_any(center: np.ndarray, walls: list[tuple[float, float, float, float]], radius: float) -> bool:
    return _common_circle_hits_any(center, walls, radius)


def _circle_rect_normal(
    center: np.ndarray,
    rect: tuple[float, float, float, float],
    radius: float,
) -> np.ndarray | None:
    return _common_circle_rect_normal(center, rect, radius)
