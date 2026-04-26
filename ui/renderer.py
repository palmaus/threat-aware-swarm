"""Рендеринг кадра в PNG для веб‑клиента без открытия окна."""

from __future__ import annotations

import io
import os
from types import SimpleNamespace

import numpy as np
import pygame
from PIL import Image

from env.visualizer import SwarmVisualizer
from ui.overlay import OverlayFlags

if os.environ.get("DISPLAY", "") == "" and os.environ.get("SDL_VIDEODRIVER") is None:
    # В безэкранном режиме pygame требует фиктивный драйвер, иначе он пытается открыть окно.
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def _surface_to_png(surface: pygame.Surface) -> bytes:
    # PNG‑кодирование отделено, чтобы не смешивать рендер и сериализацию.
    raw = pygame.image.tostring(surface, "RGB")
    img = Image.frombytes("RGB", surface.get_size(), raw)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _surface_to_array(surface: pygame.Surface) -> np.ndarray:
    # Pygame отдаёт массив в формате (W, H, C); приводим к (H, W, C).
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2)).astype(np.uint8)


def _draw_trails(viz: SwarmVisualizer, trails: list[list[tuple[float, float]]]):
    for pts in trails:
        if len(pts) < 2:
            continue
        color = (60, 60, 60)
        for i in range(1, len(pts)):
            x1, y1 = pts[i - 1]
            x2, y2 = pts[i]
            p1 = (int(x1 * viz.scale), int(viz.screen_size - y1 * viz.scale))
            p2 = (int(x2 * viz.scale), int(viz.screen_size - y2 * viz.scale))
            pygame.draw.line(viz.screen, color, p1, p2, 1)


class _RenderSimView:
    """Минимальный read-only adapter под SwarmVisualizer без доступа к env.sim."""

    def __init__(self, snapshot: dict):
        self.agents_pos = np.asarray(snapshot.get("agents_pos", np.zeros((0, 2))), dtype=np.float32)
        self.agents_active = np.asarray(snapshot.get("agents_active", np.zeros((0,), dtype=bool)), dtype=bool)
        self.n = int(self.agents_pos.shape[0])
        self.time_step = int(snapshot.get("timestep", 0))
        self.walls = [tuple(float(v) for v in wall) for wall in (snapshot.get("walls") or [])]
        threats = []
        for raw in snapshot.get("threats") or []:
            try:
                pos, radius, intensity, velocity = raw
            except Exception:
                continue
            threats.append(
                SimpleNamespace(
                    position=np.asarray(pos, dtype=np.float32),
                    radius=float(radius),
                    intensity=float(intensity),
                    velocity=np.asarray(velocity, dtype=np.float32),
                )
            )
        self.threats = threats

    def get_state(self):
        return self.agents_pos.copy(), self.agents_active.copy()


def _draw_grid_overlay(viz: SwarmVisualizer, snapshot: dict, obs: dict | None, agent_idx: int):
    if obs is None:
        return
    if not isinstance(obs, dict):
        return
    grid = obs.get("grid")
    if grid is None:
        return
    grid = np.asarray(grid, dtype=np.float32)
    if grid.ndim == 3:
        grid = grid[0]
    if grid.ndim != 2:
        return
    w = grid.shape[0]
    center = w // 2
    res = float(snapshot.get("grid_res", 1.0))
    positions = np.asarray(snapshot.get("agents_pos", np.zeros((0, 2))), dtype=np.float32)
    if agent_idx < 0 or agent_idx >= positions.shape[0]:
        return
    pos = positions[agent_idx]
    # Рисуем риск‑сетки поверх карты для визуальной отладки восприятия агента.
    for gy in range(w):
        for gx in range(w):
            val = float(grid[gy, gx])
            if val <= 0.0:
                continue
            wx = float(pos[0]) + (gx - center) * res
            wy = float(pos[1]) + (gy - center) * res
            px = int(wx * viz.scale)
            py = int(viz.screen_size - wy * viz.scale)
            size = max(2, int(res * viz.scale))
            alpha = int(40 + 160 * max(0.0, min(1.0, val)))
            cell = pygame.Surface((size, size), pygame.SRCALPHA)
            cell.fill((255, 100, 100, alpha))
            viz.screen.blit(cell, (px - size // 2, py - size // 2))


class PygameRenderer:
    def __init__(self, env_config, *, screen_size: int = 900):
        self.viz = SwarmVisualizer(env_config, screen_size=screen_size, sidebar_width=0, headless=True)
        self.screen_size = int(screen_size)

    def render(
        self,
        env,
        *,
        overlay: OverlayFlags,
        trails: list[list[tuple[float, float]]] | None = None,
        agent_idx: int = 0,
    ) -> bytes:
        surface = self._render_surface(
            env,
            overlay=overlay,
            trails=trails,
            agent_idx=agent_idx,
        )
        return _surface_to_png(surface)

    def render_array(
        self,
        env,
        *,
        overlay: OverlayFlags,
        trails: list[list[tuple[float, float]]] | None = None,
        agent_idx: int = 0,
    ) -> np.ndarray:
        surface = self._render_surface(
            env,
            overlay=overlay,
            trails=trails,
            agent_idx=agent_idx,
        )
        return _surface_to_array(surface)

    def _render_surface(
        self,
        env,
        *,
        overlay: OverlayFlags,
        trails: list[list[tuple[float, float]]] | None = None,
        agent_idx: int = 0,
    ) -> pygame.Surface:
        snapshot = env.get_runtime_snapshot(include_oracle=True)
        sim = _RenderSimView(snapshot)
        field_size = float(snapshot.get("field_size", getattr(self.viz.config, "field_size", 100.0)))
        if field_size > 0.0:
            self.viz.config.field_size = field_size
            self.viz.scale = float(self.screen_size) / field_size

        finished_mask = snapshot.get("finished", None)
        if finished_mask is not None:
            finished_mask = np.asarray(finished_mask, dtype=bool)

        raw_target = snapshot.get("target_pos", None)
        target_pos = None if raw_target is None else np.asarray(raw_target, dtype=np.float32)
        in_goal_mask = None
        if sim is not None and target_pos is not None:
            try:
                in_goal_mask = np.asarray(snapshot.get("in_goal"), dtype=bool)
            except Exception:
                try:
                    pos = sim.agents_pos
                    alive = sim.agents_active
                    d = np.linalg.norm(pos - target_pos, axis=1)
                    in_goal_mask = (d <= float(snapshot.get("goal_radius", 0.0))) & alive
                except Exception:
                    in_goal_mask = None

        grid_obs = None
        if overlay.show_grid:
            try:
                grid_obs = env.get_agent_observation(agent_idx)
            except Exception:
                grid_obs = None

        def overlay_fn(v):
            if overlay.show_trails and trails is not None:
                _draw_trails(v, trails)
            if overlay.show_grid:
                _draw_grid_overlay(v, snapshot, grid_obs, agent_idx)

        oracle_path = env.get_oracle_path() if hasattr(env, "get_oracle_path") else []

        self.viz.render(
            sim,
            target_pos,
            float(snapshot.get("goal_radius", 0.0)),
            in_goal_mask=in_goal_mask,
            finished_mask=finished_mask,
            hero_idx=agent_idx,
            show_threats=overlay.show_threats,
            oracle_path=oracle_path,
            overlay_fn=overlay_fn,
            handle_events=False,
        )
        return self.viz.screen
