import numpy as np
import pygame

from env.config import EnvConfig
from env.physics.core import PhysicsCore

# Палитра в формате RGB.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
RED = (255, 80, 80)
GREEN = (50, 200, 50)
CYAN = (0, 200, 200)
ORANGE = (255, 165, 0)
YELLOW = (255, 220, 40)
GRAY = (200, 200, 200)
DARK_GRAY = (130, 130, 130)


class SwarmVisualizer:
    """Простой визуализатор на pygame.

    Визуализатор отделен от PettingZoo, чтобы его можно было вызывать из любых скриптов.
    Он поддерживает отрисовку угроз, цели и состояния агентов с безопасными значениями по умолчанию.
    """

    def __init__(
        self,
        config: EnvConfig,
        screen_size: int = 800,
        sidebar_width: int = 0,
        *,
        headless: bool = False,
        surface: pygame.Surface | None = None,
    ):
        pygame.init()
        self.config = config
        self.screen_size = int(screen_size)  # Квадратная область карты в пикселях.
        self.sidebar_width = int(max(0, sidebar_width))
        self.scale = float(self.screen_size) / float(config.field_size)
        self.headless = bool(headless)

        if surface is not None:
            self.screen = surface
        elif self.headless:
            self.screen = pygame.Surface((self.screen_size + self.sidebar_width, self.screen_size))
        else:
            self.screen = pygame.display.set_mode((self.screen_size + self.sidebar_width, self.screen_size))
            pygame.display.set_caption("Threat-Aware Swarm")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)

    def render(
        self,
        sim: PhysicsCore,
        target_pos=None,
        goal_radius: float = 3.0,
        in_goal_mask=None,
        finished_mask=None,
        hero_idx=None,
        show_separation_radius: float | None = None,
        show_threats: bool = True,
        oracle_path=None,
        overlay_fn=None,
        handle_events: bool = True,
    ):
        if handle_events and not self.headless:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False

        self.screen.fill(WHITE)
        self._draw_grid()

        for rect in getattr(sim, "walls", []):
            x1, y1, x2, y2 = rect
            rx = int(float(x1) * self.scale)
            ry = int(self.screen_size - float(y2) * self.scale)
            rw = int(float(x2 - x1) * self.scale)
            rh = int(float(y2 - y1) * self.scale)
            pygame.draw.rect(self.screen, GRAY, pygame.Rect(rx, ry, rw, rh))

        if oracle_path:
            pts = []
            for x, y in oracle_path:
                pts.append((int(float(x) * self.scale), int(self.screen_size - float(y) * self.scale)))
            if len(pts) >= 2:
                pygame.draw.lines(self.screen, GREEN, False, pts, 2)

        if show_threats:
            for threat in getattr(sim, "threats", []):
                pos_px = (
                    int(float(threat.position[0]) * self.scale),
                    int(self.screen_size - float(threat.position[1]) * self.scale),
                )
                radius_px = int(float(threat.radius) * self.scale)

                # Прозрачность показывает относительную интенсивность угрозы.
                alpha = int(40 + 160 * float(threat.intensity))
                surface = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
                pygame.draw.circle(surface, (255, 0, 0, alpha), (radius_px, radius_px), radius_px)
                self.screen.blit(surface, (pos_px[0] - radius_px, pos_px[1] - radius_px))

                pygame.draw.circle(self.screen, RED, pos_px, radius_px, 1)

        if target_pos is not None:
            tx = int(float(target_pos[0]) * self.scale)
            ty = int(self.screen_size - float(target_pos[1]) * self.scale)

            gr_px = max(2, int(float(goal_radius) * self.scale))
            goal_surface = pygame.Surface((gr_px * 2, gr_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(goal_surface, (0, 200, 0, 70), (gr_px, gr_px), gr_px)
            self.screen.blit(goal_surface, (tx - gr_px, ty - gr_px))

            pygame.draw.circle(self.screen, GREEN, (tx, ty), gr_px, 2)
            pygame.draw.circle(self.screen, GREEN, (tx, ty), 3)

        agents_pos, active_mask = sim.get_state()
        n = sim.n

        # Рассчитываем маски, если вызывающая сторона их не дала.
        if target_pos is not None and in_goal_mask is None:
            d = np.linalg.norm(agents_pos - np.asarray(target_pos)[None, :], axis=1)
            in_goal_mask = d < float(goal_radius)
        if finished_mask is None:
            finished_mask = np.zeros(n, dtype=bool)

        # Радиус разделения рисуется только по запросу, чтобы не перегружать кадр.
        sep_px = None
        if show_separation_radius is not None:
            sep_px = int(float(show_separation_radius) * self.scale)

        for i in range(n):
            px = int(float(agents_pos[i, 0]) * self.scale)
            py = int(self.screen_size - float(agents_pos[i, 1]) * self.scale)

            if not bool(active_mask[i]):
                # Убитого агента рисуем крестом для мгновенной визуальной диагностики.
                pygame.draw.line(self.screen, BLACK, (px - 4, py - 4), (px + 4, py + 4), 2)
                pygame.draw.line(self.screen, BLACK, (px - 4, py + 4), (px + 4, py - 4), 2)
                continue

            if bool(finished_mask[i]):
                color = GREEN
                radius = 6
            elif in_goal_mask is not None and bool(in_goal_mask[i]):
                color = ORANGE
                radius = 5
            else:
                color = BLUE
                radius = 4

            if hero_idx is not None and i == int(hero_idx):
                color = YELLOW
                radius = 7

            if sep_px is not None and (in_goal_mask is None or not bool(in_goal_mask[i])):
                pygame.draw.circle(self.screen, (80, 80, 80), (px, py), sep_px, 1)

            pygame.draw.circle(self.screen, color, (px, py), radius)

        n_alive = int(np.sum(active_mask))
        n_finished = int(np.sum(finished_mask))
        n_in_goal = int(np.sum(in_goal_mask)) if in_goal_mask is not None else 0

        hud_lines = [
            f"step={sim.time_step}",
            f"alive={n_alive}/{n}",
            f"in_goal={n_in_goal}/{n}",
            f"finished={n_finished}/{n}",
        ]

        y = 8
        for line in hud_lines:
            txt = self.font.render(line, True, BLACK)
            self.screen.blit(txt, (10, y))
            y += 18

        if overlay_fn is not None:
            try:
                overlay_fn(self)
            except Exception:
                pass

        if not self.headless:
            pygame.display.flip()
            self.clock.tick(60)
        return True

    def _draw_grid(self):
        step_px = int(10 * self.scale)
        if step_px <= 0:
            return
        for x in range(0, self.screen_size, step_px):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.screen_size), 1)
        for y in range(0, self.screen_size, step_px):
            pygame.draw.line(self.screen, GRAY, (0, y), (self.screen_size, y), 1)

        # Рамка рисуется поверх сетки, чтобы граница поля читалась лучше.
        pygame.draw.rect(self.screen, DARK_GRAY, pygame.Rect(0, 0, self.screen_size, self.screen_size), 2)

    def close(self):
        pygame.quit()
