import numpy as np

from baselines.utils import predict_target_batch, velocity_tracking_action_batch


class PotentialFieldPolicy:
    def __init__(
        self,
        n_agents,
        k_att=1.0,
        k_rep=50.0,
        safety_margin=5.0,
        repulsion_power: float = 1.0,
        push_weight: float = 0.85,
        vortex_weight: float = 0.5,
        threat_predict_time: float = 0.5,
        threat_predict_inflate: float = 0.6,
        k_wall_rep=6.0,
        k_wall_follow=4.0,
        wall_influence=6.0,
        damping: float = 0.2,
        stuck_steps: int = 20,
        stuck_dist_eps: float = 0.5,
        stuck_att_scale: float = 0.4,
        stuck_rep_scale: float = 1.3,
        intercept_enabled: bool = True,
        intercept_gain: float = 0.6,
        intercept_max_time: float = 3.0,
    ):
        """
        Политика потенциальных полей.
        k_att/k_rep задают силы притяжения и отталкивания, safety_margin — запас по радиусу угроз (в метрах).
        """
        self.n_agents = n_agents
        self.k_att = k_att
        self.k_rep = k_rep
        self.safety_margin = safety_margin
        self.repulsion_power = float(repulsion_power)
        self.push_weight = float(push_weight)
        self.vortex_weight = float(vortex_weight)
        self.threat_predict_time = float(threat_predict_time)
        self.threat_predict_inflate = float(threat_predict_inflate)
        self.k_wall_rep = float(k_wall_rep)
        self.k_wall_follow = float(k_wall_follow)
        self.wall_influence = float(wall_influence)
        self.damping = float(damping)
        self.stuck_steps = int(max(0, stuck_steps))
        self.stuck_dist_eps = float(stuck_dist_eps)
        self.stuck_att_scale = float(stuck_att_scale)
        self.stuck_rep_scale = float(stuck_rep_scale)
        self.intercept_enabled = bool(intercept_enabled)
        self.intercept_gain = float(intercept_gain)
        self.intercept_max_time = float(intercept_max_time)
        self._agent_state = {}

    def reset(self, seed: int | None = None) -> None:
        self._agent_state = {}

    def get_actions(
        self,
        agents_pos: np.ndarray,
        active_mask: np.ndarray,
        target_pos: np.ndarray,
        threats: list,
        *,
        walls: list | None = None,
        target_vel: np.ndarray | list | None = None,
        max_speed: float | None = None,
        agents_vel: np.ndarray | None = None,
        max_accel: float | None = None,
        dt: float | None = None,
        drag: float | None = None,
        oracle_dir: np.ndarray | None = None,
        output_mode: str = "action",
    ):
        """
        Собирает суммарное поле из цели, угроз и стен и превращает его в действия.
        """
        n_agents = int(np.asarray(agents_pos).shape[0])
        total_force = np.zeros_like(agents_pos)
        target = np.asarray(target_pos, dtype=np.float32)
        if self.intercept_enabled and target_vel is not None:
            target = predict_target_batch(
                target,
                target_vel,
                agents_pos,
                max_speed=max_speed,
                gain=self.intercept_gain,
                max_time=self.intercept_max_time,
                enabled=self.intercept_enabled,
            )
        to_target = target - agents_pos

        dists_target = np.linalg.norm(to_target, axis=1, keepdims=True)
        to_target_norm = to_target / (dists_target + 1e-8)
        oracle_mask = None
        if oracle_dir is not None:
            try:
                oracle_arr = np.asarray(oracle_dir, dtype=np.float32)
                if oracle_arr.shape == agents_pos.shape:
                    norms = np.linalg.norm(oracle_arr, axis=1, keepdims=True)
                    oracle_mask = norms.reshape(-1) > 1e-6
                    oracle_norm = oracle_arr / (norms + 1e-8)
                    to_target_norm = np.where(oracle_mask[:, None], oracle_norm, to_target_norm)
            except Exception:
                oracle_mask = None

        att_scale = np.ones((n_agents, 1), dtype=np.float32)
        rep_scale = np.ones((n_agents, 1), dtype=np.float32)
        if self.stuck_steps > 0:
            dists_flat = dists_target.reshape(-1)
            for i in range(n_agents):
                state = self._agent_state.setdefault(i, {"best_dist": None, "since_improve": 0})
                best = state["best_dist"]
                dist = float(dists_flat[i])
                if best is None or dist < best - self.stuck_dist_eps:
                    state["best_dist"] = dist
                    state["since_improve"] = 0
                else:
                    state["since_improve"] += 1
                if state["since_improve"] >= self.stuck_steps:
                    # Ослабляем притяжение и усиливаем отталкивание, чтобы выйти из локальных минимумов.
                    att_scale[i, 0] = float(self.stuck_att_scale)
                    rep_scale[i, 0] = float(self.stuck_rep_scale)

        total_force += to_target_norm * self.k_att * att_scale

        # Отталкивание от угроз начинается заранее, до входа в радиус поражения.
        if threats:
            try:
                t_pos = np.asarray([t.position for t in threats], dtype=np.float32)
                t_rad = np.asarray([t.radius for t in threats], dtype=np.float32)
                t_dyn = np.asarray([bool(getattr(t, "is_dynamic", False)) for t in threats], dtype=bool)
                if self.threat_predict_time > 0.0:
                    t_vel = np.asarray(
                        [getattr(t, "velocity", getattr(t, "vel", np.zeros(2, dtype=np.float32))) for t in threats],
                        dtype=np.float32,
                    )
                    lead = float(self.threat_predict_time)
                    t_pos = t_pos + (t_vel * lead)
                    t_rad = t_rad + (np.linalg.norm(t_vel, axis=1) * lead * float(self.threat_predict_inflate))
                if t_pos.ndim == 2 and t_pos.shape[0] > 0:
                    diff = agents_pos[:, None, :] - t_pos[None, :, :]
                    dists = np.linalg.norm(diff, axis=2)
                    push_dir = diff / (dists[..., None] + 1e-8)
                    influence = t_rad[None, :] + float(self.safety_margin)
                    margin = np.maximum(dists - t_rad[None, :], 0.0)
                    power = max(1.0, float(self.repulsion_power))
                    repulsion_mag = np.where(dists <= influence, 1.0 / ((margin + 1e-3) ** power), 0.0)
                    if np.any(t_dyn):
                        vortex_dir = np.stack([-push_dir[..., 1], push_dir[..., 0]], axis=-1)
                        signs = np.where((np.arange(n_agents) % 2) == 0, 1.0, -1.0).astype(np.float32)
                        signs = signs[:, None, None]
                        evasion = np.where(
                            t_dyn[None, :, None],
                            (self.push_weight * push_dir) + (self.vortex_weight * vortex_dir * signs),
                            push_dir,
                        )
                        evasion = evasion / (np.linalg.norm(evasion, axis=2, keepdims=True) + 1e-8)
                        repulsion = evasion * repulsion_mag[..., None]
                    else:
                        repulsion = push_dir * repulsion_mag[..., None]
                    total_force += repulsion.sum(axis=1) * float(self.k_rep) * rep_scale
                else:
                    raise ValueError("empty threat array")
            except Exception:
                for threat in threats:
                    t_pos_single = np.asarray(threat.position, dtype=np.float32)
                    t_rad_single = float(threat.radius)
                    if self.threat_predict_time > 0.0:
                        t_vel_single = np.asarray(
                            getattr(threat, "velocity", getattr(threat, "vel", np.zeros(2, dtype=np.float32))),
                            dtype=np.float32,
                        )
                        lead = float(self.threat_predict_time)
                        t_pos_single = t_pos_single + (t_vel_single * lead)
                        t_rad_single = t_rad_single + (
                            np.linalg.norm(t_vel_single) * lead * float(self.threat_predict_inflate)
                        )

                    from_threat = agents_pos - t_pos_single
                    dists_threat = np.linalg.norm(from_threat, axis=1, keepdims=True)

                    push_dir = from_threat / (dists_threat + 1e-8)

                    # Зона влияния: радиус угрозы + запас безопасности.
                    influence_radius = float(t_rad_single) + self.safety_margin

                    margin = np.maximum(dists_threat - float(t_rad_single), 0.0)
                    power = max(1.0, float(self.repulsion_power))
                    repulsion_mag = np.where(dists_threat <= influence_radius, 1.0 / ((margin + 1e-3) ** power), 0.0)
                    if bool(getattr(threat, "is_dynamic", False)):
                        vortex_dir = np.concatenate([-push_dir[:, 1:2], push_dir[:, 0:1]], axis=1)
                        signs = np.where((np.arange(n_agents) % 2) == 0, 1.0, -1.0).astype(np.float32)[:, None]
                        evasion = (self.push_weight * push_dir) + (self.vortex_weight * vortex_dir * signs)
                        evasion = evasion / (np.linalg.norm(evasion, axis=1, keepdims=True) + 1e-8)
                        total_force += evasion * repulsion_mag * self.k_rep * rep_scale
                    else:
                        total_force += push_dir * repulsion_mag * self.k_rep * rep_scale

        # Стены дают отталкивание и касательную составляющую для скольжения.
        if walls:
            wall_rep, wall_tan = self._wall_forces_batch(agents_pos, to_target_norm, walls)
            total_force += (self.k_wall_rep * wall_rep) + (self.k_wall_follow * wall_tan)

        if agents_vel is not None and self.damping > 0.0:
            try:
                vel = np.asarray(agents_vel, dtype=np.float32)
                if vel.shape == total_force.shape:
                    total_force -= float(self.damping) * vel
            except Exception:
                pass

        # Мертвые агенты не должны влиять на общую динамику.
        total_force[~active_mask] = 0.0

        actions = self._normalize_actions(total_force)
        if (
            output_mode != "desired"
            and agents_vel is not None
            and max_speed is not None
            and max_accel is not None
            and dt is not None
        ):
            try:
                actions = velocity_tracking_action_batch(
                    actions,
                    np.asarray(agents_vel, dtype=np.float32),
                    max_speed=float(max_speed),
                    max_accel=float(max_accel),
                    dt=float(dt),
                    drag=float(drag or 0.0),
                )
            except Exception:
                pass
        return actions

    def _wall_forces(self, pos: np.ndarray, to_target_dir: np.ndarray, walls: list) -> tuple[np.ndarray, np.ndarray]:
        rep = np.zeros(2, dtype=np.float32)
        tan = np.zeros(2, dtype=np.float32)
        influence = max(self.wall_influence, 1e-6)
        for rect in walls:
            try:
                x1, y1, x2, y2 = rect
            except Exception:
                continue
            cx = float(np.clip(pos[0], x1, x2))
            cy = float(np.clip(pos[1], y1, y2))
            vec = np.array([pos[0] - cx, pos[1] - cy], dtype=np.float32)
            dist = float(np.linalg.norm(vec))
            if dist > influence:
                continue
            if dist <= 1e-6:
                # Точка внутри стены: выталкиваем наружу по ближайшей оси.
                dx = min(abs(pos[0] - x1), abs(x2 - pos[0]))
                dy = min(abs(pos[1] - y1), abs(y2 - pos[1]))
                if dx < dy:
                    vec = (
                        np.array([1.0, 0.0], dtype=np.float32)
                        if pos[0] >= (x1 + x2) / 2
                        else np.array([-1.0, 0.0], dtype=np.float32)
                    )
                else:
                    vec = (
                        np.array([0.0, 1.0], dtype=np.float32)
                        if pos[1] >= (y1 + y2) / 2
                        else np.array([0.0, -1.0], dtype=np.float32)
                    )
                dist = 1.0
            rep_dir = vec / dist
            strength = (influence - dist) / influence
            rep += rep_dir * strength
            t1 = np.array([-rep_dir[1], rep_dir[0]], dtype=np.float32)
            t2 = -t1
            if float(np.dot(t1, to_target_dir)) >= float(np.dot(t2, to_target_dir)):
                tan += t1 * strength
            else:
                tan += t2 * strength
        return rep, tan

    def _wall_forces_batch(
        self,
        pos: np.ndarray,
        to_target_dir: np.ndarray,
        walls: list,
    ) -> tuple[np.ndarray, np.ndarray]:
        rep = np.zeros_like(pos, dtype=np.float32)
        tan = np.zeros_like(pos, dtype=np.float32)
        influence = max(self.wall_influence, 1e-6)
        pos = np.asarray(pos, dtype=np.float32)
        to_target_dir = np.asarray(to_target_dir, dtype=np.float32)
        for rect in walls:
            try:
                x1, y1, x2, y2 = rect
            except Exception:
                continue
            cx = np.clip(pos[:, 0], x1, x2)
            cy = np.clip(pos[:, 1], y1, y2)
            vec = np.stack([pos[:, 0] - cx, pos[:, 1] - cy], axis=1).astype(np.float32)
            dist = np.linalg.norm(vec, axis=1)
            inside = dist <= 1e-6
            if np.any(inside):
                dx = np.minimum(np.abs(pos[:, 0] - x1), np.abs(x2 - pos[:, 0]))
                dy = np.minimum(np.abs(pos[:, 1] - y1), np.abs(y2 - pos[:, 1]))
                use_x = dx < dy
                center_x = 0.5 * (x1 + x2)
                center_y = 0.5 * (y1 + y2)
                sign_x = np.where(pos[:, 0] >= center_x, 1.0, -1.0).astype(np.float32)
                sign_y = np.where(pos[:, 1] >= center_y, 1.0, -1.0).astype(np.float32)
                mask_x = inside & use_x
                mask_y = inside & ~use_x
                vec[mask_x, 0] = sign_x[mask_x]
                vec[mask_x, 1] = 0.0
                vec[mask_y, 0] = 0.0
                vec[mask_y, 1] = sign_y[mask_y]
                dist = np.where(inside, 1.0, dist)

            rep_dir = vec / (dist[:, None] + 1e-8)
            strength = (influence - dist) / influence
            strength = np.where(dist <= influence, strength, 0.0).astype(np.float32)
            rep += rep_dir * strength[:, None]

            t1 = np.stack([-rep_dir[:, 1], rep_dir[:, 0]], axis=1)
            dot1 = np.sum(t1 * to_target_dir, axis=1)
            use_t1 = dot1 >= 0.0
            tan_dir = np.where(use_t1[:, None], t1, -t1)
            tan += tan_dir * strength[:, None]
        return rep, tan

    @staticmethod
    def _normalize_actions(actions: np.ndarray) -> np.ndarray:
        # Масштабируем в [-1, 1], сохраняя направление вектора.
        mags = np.linalg.norm(actions, axis=1, keepdims=True)
        safe = np.maximum(mags, 1e-8)
        scaled = actions / safe
        return np.clip(scaled, -1.0, 1.0)
