from __future__ import annotations

import math
from typing import Any

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    mag = float(np.linalg.norm(vec))
    if mag <= 1e-6:
        return np.zeros((2,), dtype=np.float32)
    return (vec / mag).astype(np.float32)


def _resolve_circle_rect(pos: np.ndarray, radius: float, rect: tuple[float, float, float, float]):
    x1, y1, x2, y2 = rect
    cx = float(np.clip(pos[0], x1, x2))
    cy = float(np.clip(pos[1], y1, y2))
    dx = float(pos[0] - cx)
    dy = float(pos[1] - cy)
    dist2 = dx * dx + dy * dy
    r2 = float(radius * radius)
    if dist2 >= r2:
        return None
    if dist2 > 1e-8:
        dist = math.sqrt(dist2)
        normal = np.array([dx / dist, dy / dist], dtype=np.float32)
        penetration = float(radius - dist)
        return normal, penetration

    # Центр внутри прямоугольника: выталкиваем к ближайшей стороне.
    left = abs(float(pos[0] - x1))
    right = abs(float(x2 - pos[0]))
    down = abs(float(pos[1] - y1))
    up = abs(float(y2 - pos[1]))
    min_dist = min(left, right, down, up)
    if min_dist == left:
        normal = np.array([-1.0, 0.0], dtype=np.float32)
        penetration = float(radius + left)
    elif min_dist == right:
        normal = np.array([1.0, 0.0], dtype=np.float32)
        penetration = float(radius + right)
    elif min_dist == down:
        normal = np.array([0.0, -1.0], dtype=np.float32)
        penetration = float(radius + down)
    else:
        normal = np.array([0.0, 1.0], dtype=np.float32)
        penetration = float(radius + up)
    return normal, penetration


def _move_with_bounce(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float,
    field_size: float,
    walls: list[tuple[float, float, float, float]] | None,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pos, dtype=np.float32)
    v = np.asarray(vel, dtype=np.float32)
    if dt <= 0.0:
        return p, v

    # Простая кинематика с отражением от границ и стен, без интегрирования ускорений.
    p = p + v * float(dt)

    min_pos = float(max(radius, 0.0))
    max_pos = float(max(field_size - radius, min_pos))
    for axis in (0, 1):
        if p[axis] < min_pos:
            p[axis] = min_pos
            v[axis] = abs(v[axis])
        elif p[axis] > max_pos:
            p[axis] = max_pos
            v[axis] = -abs(v[axis])

    if walls:
        for rect in walls:
            hit = _resolve_circle_rect(p, radius, rect)
            if hit is None:
                continue
            normal, penetration = hit
            p = p + normal * penetration
            vn = float(np.dot(v, normal))
            if vn < 0.0:
                v = v - (2.0 * vn) * normal
    return p.astype(np.float32), v.astype(np.float32)


def _copy_radius_dynamics(src, dst) -> None:
    for attr in (
        "radius_min",
        "radius_max",
        "radius_speed",
        "radius_mode",
        "radius_noise",
        "_radius_phase",
        "_radius_vel",
    ):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _apply_radius_dynamics(threat, cfg: dict[str, Any], rng: np.random.Generator) -> None:
    rmin = None
    rmax = None
    if "radius_range" in cfg:
        try:
            rrange = list(cfg.get("radius_range") or [])
            if len(rrange) >= 2:
                rmin = rrange[0]
                rmax = rrange[1]
        except Exception:
            rmin = None
            rmax = None
    if rmin is None:
        rmin = cfg.get("radius_min") or cfg.get("min_radius") or cfg.get("r_min")
    if rmax is None:
        rmax = cfg.get("radius_max") or cfg.get("max_radius") or cfg.get("r_max")
    if rmin is None or rmax is None:
        return
    rmin = float(rmin)
    rmax = float(rmax)
    if rmax < rmin:
        rmin, rmax = rmax, rmin
    # Динамика радиуса делает угрозу «дышащей» и требует более осторожного поведения.
    threat.radius_min = rmin
    threat.radius_max = rmax
    threat.radius_mode = str(cfg.get("radius_mode", "sine")).lower()
    threat.radius_speed = float(cfg.get("radius_speed", cfg.get("radius_freq", 0.5)))
    threat.radius_noise = float(cfg.get("radius_noise", cfg.get("radius_jitter", 0.2)))
    phase = cfg.get("radius_phase")
    if phase is None:
        phase = rng.uniform(0.0, 2.0 * math.pi)
    threat._radius_phase = float(phase)
    threat._radius_vel = 0.0
    threat.radius = float(np.clip(float(threat.radius), rmin, rmax))


def _ensure_rng(rng: np.random.Generator | np.random.RandomState | None) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, np.random.RandomState):
        seed = int(rng.randint(0, 2**32 - 1))
        return np.random.default_rng(seed)
    return np.random.default_rng()


def _copy_rng(rng: np.random.Generator | None) -> np.random.Generator:
    new_rng = np.random.default_rng()
    if rng is None:
        return new_rng
    try:
        new_rng.bit_generator.state = rng.bit_generator.state
    except Exception:
        pass
    return new_rng


class BaseThreat:
    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        intensity: float,
        *,
        oracle_block: bool = False,
        kind: str = "static",
        rng: np.random.Generator | np.random.RandomState | None = None,
    ):
        self.position = np.asarray(position, dtype=np.float32)
        self.radius = float(radius)
        self.intensity = float(intensity)
        self.oracle_block = bool(oracle_block)
        self.kind = str(kind)
        self.rng = _ensure_rng(rng)

    @property
    def is_dynamic(self) -> bool:
        return False

    def update(
        self,
        dt: float,
        field_size: float,
        walls: list[tuple[float, float, float, float]] | None,
        agents_pos: np.ndarray | None = None,
        agents_active: np.ndarray | None = None,
    ) -> None:
        return None

    def get_state(self) -> dict[str, Any]:
        return {
            "pos": self.position.copy(),
            "radius": float(self.radius),
            "intensity": float(self.intensity),
            "oracle_block": bool(self.oracle_block),
            "kind": str(self.kind),
        }

    def copy(self):
        clone = BaseThreat(
            self.position.copy(),
            self.radius,
            self.intensity,
            oracle_block=self.oracle_block,
            kind=self.kind,
            rng=_copy_rng(self.rng),
        )
        _copy_radius_dynamics(self, clone)
        return clone


class StaticThreat(BaseThreat):
    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        intensity: float,
        *,
        oracle_block: bool = False,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ):
        super().__init__(position, radius, intensity, oracle_block=oracle_block, kind="static", rng=rng)

    def copy(self):
        clone = StaticThreat(
            self.position.copy(),
            self.radius,
            self.intensity,
            oracle_block=self.oracle_block,
            rng=_copy_rng(self.rng),
        )
        _copy_radius_dynamics(self, clone)
        return clone


class DynamicThreat(BaseThreat):
    @property
    def is_dynamic(self) -> bool:
        return True

    def _update_radius(self, dt: float) -> None:
        if dt <= 0.0:
            return
        if not hasattr(self, "radius_min") or not hasattr(self, "radius_max"):
            return
        rmin = getattr(self, "radius_min", None)
        rmax = getattr(self, "radius_max", None)
        if rmin is None or rmax is None:
            return
        rmin = float(rmin)
        rmax = float(rmax)
        if rmax <= rmin:
            self.radius = rmin
            return
        mode = str(getattr(self, "radius_mode", "sine")).lower()
        speed = float(getattr(self, "radius_speed", 0.5))
        if mode in {"sine", "sin", "cos"}:
            # Синусоидальный режим предсказуем и подходит для обучающего сигнала.
            phase = float(getattr(self, "_radius_phase", 0.0))
            phase += float(speed) * float(dt) * 2.0 * math.pi
            self._radius_phase = phase
            base = 0.5 * (rmin + rmax)
            amp = 0.5 * (rmax - rmin)
            self.radius = base + amp * math.sin(phase)
        elif mode in {"random", "noise", "rw"}:
            # Случайный режим добавляет неопределённость, но остаётся плавным.
            vel = float(getattr(self, "_radius_vel", 0.0))
            noise = float(getattr(self, "radius_noise", 0.2))
            vel += float(self.rng.normal(0.0, noise)) * float(dt)
            vel = float(np.clip(vel, -abs(speed), abs(speed)))
            self._radius_vel = vel
            self.radius = float(np.clip(self.radius + vel * dt, rmin, rmax))
        else:
            self.radius = float(np.clip(self.radius, rmin, rmax))


class LinearThreat(DynamicThreat):
    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        intensity: float,
        velocity: np.ndarray,
        *,
        oracle_block: bool = False,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ):
        super().__init__(position, radius, intensity, oracle_block=oracle_block, kind="linear", rng=rng)
        self.velocity = np.asarray(velocity, dtype=np.float32)

    def update(
        self,
        dt: float,
        field_size: float,
        walls,
        agents_pos: np.ndarray | None = None,
        agents_active: np.ndarray | None = None,
    ) -> None:
        self._update_radius(dt)
        # Линейные угрозы отражаются от границ и стен, сохраняя скорость.
        self.position, self.velocity = _move_with_bounce(
            self.position,
            self.velocity,
            dt,
            field_size,
            walls,
            self.radius,
        )

    def copy(self):
        clone = LinearThreat(
            self.position.copy(),
            self.radius,
            self.intensity,
            self.velocity.copy(),
            oracle_block=self.oracle_block,
            rng=_copy_rng(self.rng),
        )
        _copy_radius_dynamics(self, clone)
        return clone


class BrownianThreat(DynamicThreat):
    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        intensity: float,
        speed: float,
        noise_scale: float,
        *,
        oracle_block: bool = False,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ):
        super().__init__(position, radius, intensity, oracle_block=oracle_block, kind="brownian", rng=rng)
        self.speed = float(speed)
        self.noise_scale = float(noise_scale)
        self.rng = _ensure_rng(rng)
        self.velocity = _normalize(self.rng.normal(size=2).astype(np.float32)) * float(self.speed)

    def update(
        self,
        dt: float,
        field_size: float,
        walls,
        agents_pos: np.ndarray | None = None,
        agents_active: np.ndarray | None = None,
    ) -> None:
        self._update_radius(dt)
        # Броуновское движение добавляет шум направления для непредсказуемости.
        jitter = self.rng.normal(0.0, self.noise_scale, size=2).astype(np.float32)
        direction = _normalize(self.velocity + jitter)
        if float(np.linalg.norm(direction)) <= 1e-6:
            direction = _normalize(self.rng.normal(size=2).astype(np.float32))
        self.velocity = direction * float(self.speed)
        self.position, self.velocity = _move_with_bounce(
            self.position,
            self.velocity,
            dt,
            field_size,
            walls,
            self.radius,
        )

    def copy(self):
        rng = _copy_rng(self.rng)
        clone = BrownianThreat(
            self.position.copy(),
            self.radius,
            self.intensity,
            self.speed,
            self.noise_scale,
            oracle_block=self.oracle_block,
            rng=rng,
        )
        clone.velocity = self.velocity.copy()
        _copy_radius_dynamics(self, clone)
        return clone


class ChaserThreat(DynamicThreat):
    def __init__(
        self,
        position: np.ndarray,
        radius: float,
        intensity: float,
        speed: float,
        vision_radius: float,
        *,
        oracle_block: bool = False,
        rng: np.random.Generator | np.random.RandomState | None = None,
    ):
        super().__init__(position, radius, intensity, oracle_block=oracle_block, kind="chaser", rng=rng)
        self.speed = float(speed)
        self.vision_radius = float(vision_radius)
        self.velocity = np.zeros((2,), dtype=np.float32)

    def update(
        self,
        dt: float,
        field_size: float,
        walls,
        agents_pos: np.ndarray | None = None,
        agents_active: np.ndarray | None = None,
    ) -> None:
        self._update_radius(dt)
        target_positions = None
        if agents_pos is not None and len(agents_pos) > 0:
            try:
                positions = np.asarray(agents_pos, dtype=np.float32)
            except Exception:
                positions = None
            if positions is not None:
                if agents_active is not None:
                    mask = np.asarray(agents_active, dtype=bool)
                    if mask.shape[0] == positions.shape[0]:
                        if np.any(mask):
                            target_positions = positions[mask]
                        else:
                            target_positions = None
                    else:
                        target_positions = positions
                else:
                    target_positions = positions

        if target_positions is not None and len(target_positions) > 0:
            diffs = target_positions - self.position[None, :]
            dists = np.linalg.norm(diffs, axis=1)
            idx = -1
            if dists.size:
                # При равных расстояниях выбираем цель детерминированно по координатам,
                # чтобы избежать случайного дрейфа порядка агентов.
                min_dist = float(np.min(dists))
                cand = np.where(dists <= (min_dist + 1e-6))[0]
                if cand.size == 1:
                    idx = int(cand[0])
                elif cand.size > 1:
                    coords = target_positions[cand]
                    order = np.lexsort((coords[:, 1], coords[:, 0]))
                    idx = int(cand[order[0]])
            if idx >= 0 and float(dists[idx]) <= self.vision_radius:
                # Преследуем ближайшего агента в зоне видимости.
                direction = _normalize(diffs[idx])
                self.velocity = direction * float(self.speed)
            else:
                # Вне видимости медленно «затухаем», чтобы не метаться.
                self.velocity = self.velocity * 0.8
        else:
            # При отсутствии целей скорость постепенно затухает.
            self.velocity = self.velocity * 0.8
        self.position, self.velocity = _move_with_bounce(
            self.position,
            self.velocity,
            dt,
            field_size,
            walls,
            self.radius,
        )

    def copy(self):
        clone = ChaserThreat(
            self.position.copy(),
            self.radius,
            self.intensity,
            self.speed,
            self.vision_radius,
            oracle_block=self.oracle_block,
            rng=_copy_rng(self.rng),
        )
        clone.velocity = self.velocity.copy()
        _copy_radius_dynamics(self, clone)
        return clone


def _parse_velocity(cfg: dict, rng: np.random.Generator, default_speed: float) -> np.ndarray:
    if "vel" in cfg:
        return np.asarray(cfg["vel"], dtype=np.float32)
    if "velocity" in cfg:
        return np.asarray(cfg["velocity"], dtype=np.float32)
    speed = float(cfg.get("speed", default_speed))
    if "direction" in cfg:
        vec = np.asarray(cfg["direction"], dtype=np.float32)
        return _normalize(vec) * speed
    if "angle" in cfg:
        ang = float(cfg["angle"])
        vec = np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))], dtype=np.float32)
        return _normalize(vec) * speed
    vec = rng.normal(size=2).astype(np.float32)
    return _normalize(vec) * speed


def threat_from_config(
    cfg: dict[str, Any], rng: np.random.Generator | np.random.RandomState | None = None
) -> BaseThreat:
    rng = _ensure_rng(rng)
    t_type = str(cfg.get("type") or cfg.get("behavior") or "static").lower()
    pos = cfg.get("pos", None)
    if pos is None:
        pos = cfg.get("position", None)
    if pos is None:
        pos = cfg.get("center", None)
    if pos is None:
        pos = [0.0, 0.0]
    radius = float(cfg.get("radius", 1.0))
    intensity = float(cfg.get("intensity", 0.1))
    # Угрозы, блокирующие оракул, считаются «статикой» для планировщика.
    # По умолчанию все статические угрозы блокируют оракул, если явно не указано иначе.
    if "oracle_block" in cfg:
        oracle_block = bool(cfg.get("oracle_block"))
    else:
        oracle_block = bool(cfg.get("static") or cfg.get("is_static") or t_type == "static")

    if t_type in {"linear", "ballistic"}:
        vel = _parse_velocity(cfg, rng, default_speed=3.0)
        threat = LinearThreat(pos, radius, intensity, vel, oracle_block=oracle_block, rng=rng)
        _apply_radius_dynamics(threat, cfg, rng)
        return threat
    if t_type in {"brownian", "stochastic", "random_walk"}:
        speed = float(cfg.get("speed", 2.0))
        noise_scale = float(cfg.get("noise_scale", 0.5))
        threat = BrownianThreat(
            pos,
            radius,
            intensity,
            speed=speed,
            noise_scale=noise_scale,
            oracle_block=oracle_block,
            rng=rng,
        )
        _apply_radius_dynamics(threat, cfg, rng)
        return threat
    if t_type in {"chaser", "pursuer", "hunter"}:
        speed = float(cfg.get("speed", 1.5))
        vision = float(cfg.get("vision_radius", 30.0))
        threat = ChaserThreat(
            pos,
            radius,
            intensity,
            speed=speed,
            vision_radius=vision,
            oracle_block=oracle_block,
            rng=rng,
        )
        _apply_radius_dynamics(threat, cfg, rng)
        return threat

    threat = StaticThreat(pos, radius, intensity, oracle_block=oracle_block, rng=rng)
    _apply_radius_dynamics(threat, cfg, rng)
    return threat


def is_dynamic_threat(obj: BaseThreat) -> bool:
    return bool(getattr(obj, "is_dynamic", False))
