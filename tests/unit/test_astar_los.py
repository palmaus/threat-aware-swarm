import numpy as np

from baselines.astar_grid import AStarGridPolicy


def test_los_skips_intermediate_cells():
    policy = AStarGridPolicy(los_enabled=True, los_max_skip=10)
    wall_mask = np.zeros((5, 5), dtype=bool)
    path = [(2, 2), (3, 2), (4, 2)]
    nxt = policy._select_los_waypoint(path, (2, 2), wall_mask)
    assert nxt == (4, 2)


def test_los_respects_walls():
    policy = AStarGridPolicy(los_enabled=True, los_max_skip=10)
    wall_mask = np.zeros((5, 5), dtype=bool)
    wall_mask[2, 3] = True
    path = [(2, 2), (2, 3), (3, 3), (4, 3), (4, 2)]
    nxt = policy._select_los_waypoint(path, (2, 2), wall_mask)
    assert nxt == (3, 3)
