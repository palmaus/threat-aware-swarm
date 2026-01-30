"""PettingZoo ParallelEnv для задачи роя, учитывающего угрозы.

Эта среда предназначена для стека:
  PettingZoo (Parallel API) + SuperSuit + Stable-Baselines3 (PPO)

Основные особенности реализации:
1) Глобальное завершение эпизода:
   Эпизод завершается для *всех* агентов одновременно (по таймауту / все мертвы / все завершили).

2) Мертвые / завершившие агенты остаются:
   Набор агентов (possible_agents) фиксирован в течение эпизода.
   Если агент умирает, он получает штраф за смерть, затем нулевую награду и наблюдение.
   Если агент завершает задачу, его действия игнорируются.

3) Информация для логирования:
   Каждый шаг возвращает информацию по каждому агенту, включая:
     расстояние, живой, в цели, завершил, риск, минимальное расстояние до соседа.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from env.config import EnvConfig
from env.core import SwarmSimulator


class RewardConfig:
    """Настройки формирования наград.
    Значения по умолчанию стабильны для PPO.
    """

    # Основная задача
    w_progress: float = 5.0            # вес за прогресс (уменьшение расстояния до цели)
    w_finish_bonus: float = 50.0       # бонус за завершение задачи
    w_in_goal_step: float = 1.0        # награда за нахождение в цели
    w_center: float = 1.0              # дополнительная награда за приближение к центру цели

    # Безопасность / ограничения
    death_penalty: float = 200.0       # штраф за смерть
    w_risk: float = 2.0                # штраф за риск
    w_wall: float = 20.0               # штраф за приближение к стенам

    # Регуляризация движения
    w_speed: float = 0.05              # штраф за скорость
    brake_dist: float = 10.0           # дистанция начала торможения

    # Расстояние между агентами
    sep_radius: float = 1.5            # минимальное расстояние до соседа
    w_sep: float = 0.5                 # штраф за близость к соседу
    sep_disable_in_goal: bool = True   # не штрафовать за близость в зоне цели


class SwarmPZEnv(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "swarm_v1"}

    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode=None,
        reward_cfg: RewardConfig | None = None,
        *,
        goal_radius: float = 3.0,
        goal_hold_steps: int = 10,
        max_steps: int = 600,
    ):
        if config is None:
            config = EnvConfig()
        self.config = config
        self.sim = SwarmSimulator(config)
        self.render_mode = render_mode

        self.reward = reward_cfg if reward_cfg is not None else RewardConfig()

        # Agents
        self.possible_agents = [f"drone_{i}" for i in range(config.n_agents)]
        self.agents = self.possible_agents[:]  # keep fixed set for vectorization

        # Task geometry
        self.goal_radius = float(goal_radius)
        self.goal_hold_steps = int(goal_hold_steps)
        self.max_steps = int(max_steps)

        # Observation layout
        # [to_target(2), vel(2), walls(4), risk_grid(11x11)]
        self.grid_width = 11
        # Base observation: to_target(2) + vel(2) + wall_lidar(4) + risk_grid(grid_width^2)
        self.base_obs_dim = 2 + 2 + 4 + (self.grid_width**2)
        # Some older checkpoints were trained with extra scalars appended.
        # To evaluate them reliably, we can set obs_dim_target to match the checkpoint's expected dim.
        self.obs_dim = int(self.base_obs_dim)

        self.observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Episode state
        self.target_pos = np.zeros(2, dtype=np.float32)
        self.prev_dists = np.zeros(config.n_agents, dtype=np.float32)
        self.was_alive = np.ones(config.n_agents, dtype=bool)
        self.in_goal_steps = np.zeros(config.n_agents, dtype=np.int32)
        self.finished = np.zeros(config.n_agents, dtype=bool)

    # --------- PettingZoo API ---------

    def reset(self, seed=None, options=None):
        # Сброс состояния среды и агентов
        self.agents = self.possible_agents[:]

        self.sim.reset()

        # Случайная инициализация цели и начальных позиций
        self.target_pos = np.random.uniform(80, 95, 2).astype(np.float32)
        start_center = np.random.uniform(5, 20, 2).astype(np.float32)
        self.sim.agents_pos = start_center + np.random.normal(0, 2, (self.config.n_agents, 2)).astype(np.float32)
        self.sim.agents_pos = np.clip(self.sim.agents_pos, 0, self.config.field_size)

        # Генерация угроз
        for _ in range(np.random.randint(3, 6)):
            t_pos = np.random.uniform(30, 70, 2).astype(np.float32)
            self.sim.add_threat(t_pos, np.random.uniform(10, 15), 0.1)

        self.prev_dists = np.linalg.norm(self.sim.agents_pos - self.target_pos, axis=1).astype(np.float32)
        self.was_alive[:] = True
        self.in_goal_steps[:] = 0
        self.finished[:] = False

        observations = {a: self._get_obs(i) for i, a in enumerate(self.possible_agents)}
        infos = {a: {} for a in self.possible_agents}
        return observations, infos

    def step(self, actions):
        # Применение действий для всех агентов и выполнение одного шага симуляции

        # Кэширование состояния агентов до шага симуляции
        alive_before = self.sim.agents_active.copy()

        # Построение команд скоростей (мертвые/завершившие агенты получают нулевую скорость)
        velocities = np.zeros((self.sim.n, 2), dtype=np.float32)
        for agent_id, action in (actions or {}).items():
            try:
                idx = int(agent_id.split("_")[1])
            except Exception:
                continue

            if idx < 0 or idx >= self.sim.n:
                continue

            if not self.sim.agents_active[idx]:
                continue
            if self.finished[idx]:
                continue

            act = np.asarray(action, dtype=np.float32)
            velocities[idx] = act * float(self.config.max_speed)

        # Выполнение одного шага физической симуляции
        self.sim.step(velocities)

        # Пост-обработка после шага
        pos = self.sim.agents_pos
        vel = self.sim.agents_vel
        alive = self.sim.agents_active
        dists = np.linalg.norm(pos - self.target_pos, axis=1).astype(np.float32)

        # Вычисление сигналов для каждого агента
        in_goal = (dists <= self.goal_radius) & alive

        # Логика удержания в цели
        self.in_goal_steps[in_goal] += 1
        self.in_goal_steps[~in_goal] = 0

        newly_finished = (~self.finished) & (self.in_goal_steps >= self.goal_hold_steps)
        self.finished[newly_finished] = True

        # Вероятность риска для каждого агента
        risk_p = self._compute_risk_probs(pos)

        # Минимальное расстояние до ближайшего соседа
        min_neighbor_dist = self._compute_min_neighbor_dist(pos, alive)

        # Условие завершения эпизода
        is_timeout = self.sim.time_step >= self.max_steps
        all_dead = not bool(np.any(alive))
        active_mask = alive
        all_finished = bool(np.all(self.finished[active_mask])) if np.any(active_mask) else False
        done = bool(is_timeout or all_dead or all_finished)

        # Награды и выходные данные
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        # Обнаружение смертей для одноразового штрафа
        died_this_step = alive_before & (~alive)

        for i, agent_id in enumerate(self.possible_agents):
            # Минимальное расстояние до соседа имеет смысл только для живых агентов
            if not alive[i]:
                mnd = 0.0
            else:
                mnd = float(min_neighbor_dist[i])
                if not np.isfinite(mnd):
                    # Если это последний живой агент, считать его очень безопасным, а не 0.
                    mnd = float(self.config.field_size)

            # Информация для логирования (всегда включать ключи; это ожидается в колбэке)
            infos[agent_id] = {
                "dist": float(dists[i]),
                "alive": float(alive[i]),
                "in_goal": float(in_goal[i]),
                "finished": float(self.finished[i]),
                "finished_alive": float(bool(self.finished[i]) and bool(alive[i])),
                "in_goal_steps": int(self.in_goal_steps[i]),
                "newly_finished": float(newly_finished[i]),
                "risk_p": float(risk_p[i]),
                "min_neighbor_dist": mnd,
            }

            # Наблюдения
            if not alive[i]:
                observations[agent_id] = np.zeros(self.obs_dim, dtype=np.float32)
            else:
                observations[agent_id] = self._get_obs(i)

            # Награды
            rew = 0.0

            # Одноразовый штраф за смерть
            if died_this_step[i]:
                rew -= float(self.reward.death_penalty)

            # Формирование наград для живых агентов
            if alive[i] and (not self.finished[i]):
                # (1) Награда за прогресс
                rew += float(self.reward.w_progress) * float(self.prev_dists[i] - dists[i])

                # (2) Награда за нахождение в цели + приближение к центру
                if in_goal[i]:
                    rew += float(self.reward.w_in_goal_step)
                    # Поощрение достижения центра (предотвращает остановку на границе)
                    center_bonus = max(0.0, 1.0 - float(dists[i] / max(self.goal_radius, 1e-6)))
                    rew += float(self.reward.w_center) * center_bonus

                # (3) Бонус за завершение (только один раз)
                if newly_finished[i]:
                    rew += float(self.reward.w_finish_bonus)

                # (4) Штраф за риск
                rew -= float(self.reward.w_risk) * float(risk_p[i])

                # (5) Штраф за приближение к стенам
                walls = self.sim.get_wall_distances(i)
                min_wall = float(np.min(walls))
                if min_wall < 0.05:
                    rew -= float(self.reward.w_wall) * (0.05 - min_wall)

                # (6) Штраф за торможение рядом с целью (с градиентом)
                if dists[i] < float(self.reward.brake_dist):
                    speed = float(np.linalg.norm(vel[i]))
                    gate = max(0.0, (float(self.reward.brake_dist) - float(dists[i])) / max(float(self.reward.brake_dist), 1e-6))
                    rew -= float(self.reward.w_speed) * speed * gate

                # (7) Штраф за близость к соседу (отключается в зоне цели, если указано)
                if float(self.reward.w_sep) > 0.0:
                    if not (self.reward.sep_disable_in_goal and in_goal[i]):
                        md = float(min_neighbor_dist[i])
                        if np.isfinite(md) and md < float(self.reward.sep_radius):
                            # Штраф: 0, если md >= sep_radius
                            rew -= float(self.reward.w_sep) * (float(self.reward.sep_radius) - md) / max(float(self.reward.sep_radius), 1e-6)

            # Мертвые или завершившие агенты: без формирования наград (кроме одноразовых штрафов/бонусов выше)
            rewards[agent_id] = float(rew)

            # Глобальное завершение для всех агентов
            terminations[agent_id] = bool(done and (not is_timeout))
            truncations[agent_id] = bool(done and is_timeout)

        # Обновление состояния для следующего шага
        self.prev_dists = dists

        return observations, rewards, terminations, truncations, infos

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _compute_risk_probs(self, positions: np.ndarray) -> np.ndarray:
        # Моментальная вероятность смерти на каждом шаге для каждой позиции агента
        n = positions.shape[0]
        risk = np.zeros(n, dtype=np.float32)
        if not self.sim.threats:
            return risk

        # Для каждого агента, p = 1 - Π(1 - intensity_k) по угрозам, которые его покрывают
        for i in range(n):
            if not self.sim.agents_active[i]:
                continue
            prod = 1.0
            p = positions[i]
            for t in self.sim.threats:
                # внутри радиуса угрозы?
                if float(np.linalg.norm(p - t.position)) <= float(t.radius):
                    prod *= (1.0 - float(t.intensity))
            risk[i] = float(1.0 - prod)
        return risk

    def _compute_min_neighbor_dist(self, positions: np.ndarray, alive: np.ndarray) -> np.ndarray:
        #Минимальное расстояние до любого другого живого агента. Мертвые агенты -> бесконечность.
        n = positions.shape[0]
        out = np.full(n, np.inf, dtype=np.float32)
        idx = np.where(alive)[0]
        if idx.size <= 1:
            return out

        # Парные расстояния среди живых агентов (N_alive x N_alive)
        pts = positions[idx]
        diff = pts[:, None, :] - pts[None, :, :]
        dmat = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dmat, np.inf)
        mins = np.min(dmat, axis=1)
        out[idx] = mins.astype(np.float32)
        return out

    def _get_obs(self, idx: int) -> np.ndarray:
        if not self.sim.agents_active[idx]:
            return np.zeros(self.obs_dim, dtype=np.float32)

        me_pos = self.sim.agents_pos[idx]
        me_vel = self.sim.agents_vel[idx]

        # Нормализация к диапазону примерно [-1, 1]
        to_target = (self.target_pos - me_pos) / float(self.config.field_size)
        norm_vel = me_vel / max(float(self.config.max_speed), 1e-6)
        walls = self.sim.get_wall_distances(idx)  # уже нормализованы [0, 1]

        # Локальная сетка риска
        grid = np.zeros((self.grid_width, self.grid_width), dtype=np.float32)
        center = self.grid_width // 2
        res = 4.0  # метров на ячейку (фиксировано)

        for t in self.sim.threats:
            rel = t.position - me_pos
            if abs(float(rel[0])) > 25.0 or abs(float(rel[1])) > 25.0:
                continue

            cx = center + int(float(rel[0]) / res)
            cy = center + int(float(rel[1]) / res)
            r_cells = int(float(t.radius) / res)

            x1 = max(0, cx - r_cells)
            x2 = min(self.grid_width, cx + r_cells + 1)
            y1 = max(0, cy - r_cells)
            y2 = min(self.grid_width, cy + r_cells + 1)

            if x1 < x2 and y1 < y2:
                grid[y1:y2, x1:x2] = np.maximum(grid[y1:y2, x1:x2], float(t.intensity))

        obs = np.concatenate([to_target, norm_vel, walls, grid.flatten()]).astype(np.float32)
        return obs
