import pygame
import numpy as np
from env.config import EnvConfig
from env.core import SwarmSimulator

# Цвета (R, G, B)
WHITE = (255, 255, 255)  # Белый
BLACK = (0, 0, 0)  # Чёрный
BLUE = (50, 100, 255)  # Синий
RED = (255, 80, 80)  # Красный
GREEN = (50, 200, 50)  # Зелёный
CYAN = (0, 200, 200)  # Голубой
ORANGE = (255, 165, 0)  # Оранжевый
YELLOW = (255, 220, 40)  # Жёлтый
GRAY = (200, 200, 200)  # Серый
DARK_GRAY = (130, 130, 130)  # Тёмно-серый


class SwarmVisualizer:
    """Простой визуализатор на основе pygame.

    Визуализатор намеренно отделён от обёрток PettingZoo.
    Он может отображать:
      - угрозы в виде полупрозрачных дисков
      - целевую область (круг)
      - состояния агентов: активен / уничтожен / в цели / завершил

    Все опциональные аргументы имеют безопасные значения по умолчанию,
    поэтому старые скрипты продолжают работать.
    """

    def __init__(self, config: EnvConfig, screen_size: int = 800):
        pygame.init()
        self.config = config
        self.screen_size = int(screen_size)
        self.scale = float(screen_size) / float(config.field_size)

        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Threat-Aware Swarm")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)

    def render(
        self,
        sim: SwarmSimulator,
        target_pos=None,
        goal_radius: float = 3.0,
        in_goal_mask=None,
        finished_mask=None,
        hero_idx=None,
        show_separation_radius: float | None = None,
    ):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(WHITE)
        self._draw_grid()

        # Угрозы
        for threat in getattr(sim, "threats", []):
            pos_px = (
                int(float(threat.position[0]) * self.scale),
                int(self.screen_size - float(threat.position[1]) * self.scale),
            )
            radius_px = int(float(threat.radius) * self.scale)

            # Прозрачность пропорциональна интенсивности
            alpha = int(40 + 160 * float(threat.intensity))
            surface = pygame.Surface((radius_px * 2, radius_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (255, 0, 0, alpha), (radius_px, radius_px), radius_px)
            self.screen.blit(surface, (pos_px[0] - radius_px, pos_px[1] - radius_px))

            # Контур
            pygame.draw.circle(self.screen, RED, pos_px, radius_px, 1)

        # Цель / Область цели
        if target_pos is not None:
            tx = int(float(target_pos[0]) * self.scale)
            ty = int(self.screen_size - float(target_pos[1]) * self.scale)

            gr_px = max(2, int(float(goal_radius) * self.scale))
            # Полупрозрачный диск цели
            goal_surface = pygame.Surface((gr_px * 2, gr_px * 2), pygame.SRCALPHA)
            pygame.draw.circle(goal_surface, (0, 200, 0, 70), (gr_px, gr_px), gr_px)
            self.screen.blit(goal_surface, (tx - gr_px, ty - gr_px))

            # Контур цели + центральная точка
            pygame.draw.circle(self.screen, GREEN, (tx, ty), gr_px, 2)
            pygame.draw.circle(self.screen, GREEN, (tx, ty), 3)

        agents_pos, active_mask = sim.get_state()
        n = sim.n

        # Вычисление масок, если они не предоставлены
        if target_pos is not None and in_goal_mask is None:
            d = np.linalg.norm(agents_pos - np.asarray(target_pos)[None, :], axis=1)
            in_goal_mask = d < float(goal_radius)
        if finished_mask is None:
            finished_mask = np.zeros(n, dtype=bool)

        # Опциональная визуализация радиуса разделения
        sep_px = None
        if show_separation_radius is not None:
            sep_px = int(float(show_separation_radius) * self.scale)

        # Агенты
        for i in range(n):
            px = int(float(agents_pos[i, 0]) * self.scale)
            py = int(self.screen_size - float(agents_pos[i, 1]) * self.scale)

            if not bool(active_mask[i]):
                # Убит -> чёрный крест
                pygame.draw.line(self.screen, BLACK, (px - 4, py - 4), (px + 4, py + 4), 2)
                pygame.draw.line(self.screen, BLACK, (px - 4, py + 4), (px + 4, py - 4), 2)
                continue

            # Цвет в зависимости от состояния
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

        # HUD (индикаторы состояния)
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

        # Граница
        pygame.draw.rect(self.screen, DARK_GRAY, pygame.Rect(0, 0, self.screen_size, self.screen_size), 2)

    def close(self):
        pygame.quit()
