"""Проверка shared wall-memory в A*."""

import numpy as np
import pytest

import baselines.astar_grid as astar_mod
from baselines.astar_grid import AStarGridPolicy
from env.state import SimState


def _make_obs(grid: np.ndarray) -> dict[str, np.ndarray]:
    vec = np.zeros((13,), dtype=np.float32)
    return {"vector": vec, "grid": np.asarray(grid, dtype=np.float32)}


def _make_state(positions: np.ndarray, target: np.ndarray) -> SimState:
    pos = np.asarray(positions, dtype=np.float32)
    n = pos.shape[0]
    vel = np.zeros_like(pos, dtype=np.float32)
    alive = np.ones((n,), dtype=bool)
    target = np.asarray(target, dtype=np.float32)
    dists = np.linalg.norm(pos - target[None, :], axis=1).astype(np.float32)
    in_goal = np.zeros((n,), dtype=bool)
    in_goal_steps = np.zeros((n,), dtype=np.int32)
    finished = np.zeros((n,), dtype=bool)
    newly_finished = np.zeros((n,), dtype=bool)
    risk_p = np.zeros((n,), dtype=np.float32)
    min_neighbor = np.full((n,), np.inf, dtype=np.float32)
    last_action = np.zeros((n, 2), dtype=np.float32)
    walls = np.zeros((n, 4), dtype=np.float32)
    measured_accel = np.zeros_like(pos, dtype=np.float32)
    energy = np.full((n,), 100.0, dtype=np.float32)
    energy_level = np.ones((n,), dtype=np.float32)
    agent_state = np.zeros((n,), dtype=np.int8)
    return SimState(
        pos=pos,
        vel=vel,
        alive=alive,
        target_pos=target,
        target_vel=np.zeros((2,), dtype=np.float32),
        timestep=0,
        threats=[],
        dists=dists,
        in_goal=in_goal,
        in_goal_steps=in_goal_steps,
        finished=finished,
        newly_finished=newly_finished,
        risk_p=risk_p,
        min_neighbor_dist=min_neighbor,
        last_action=last_action,
        walls=walls,
        oracle_dir=None,
        static_walls=[],
        static_circles=[],
        collision_speed=np.zeros((n,), dtype=np.float32),
        measured_accel=measured_accel,
        energy=energy,
        energy_level=energy_level,
        agent_state=agent_state,
        field_size=100.0,
        control_mode="waypoint",
        max_speed=5.0,
        max_accel=5.0,
        max_thrust=5.0,
        mass=1.0,
        drag_coeff=0.0,
        dt=0.1,
        drag=0.0,
        grid_res=1.0,
        agent_radius=0.5,
        wall_friction=0.0,
    )


def test_astar_shared_wall_memory_propagates():
    policy = AStarGridPolicy(
        memory_shared=True,
        memory_walls_enabled=True,
        memory_threats_enabled=False,
        stuck_steps=0,
        global_plan_enabled=False,
        escape_steps=0,
    )
    grid = np.zeros((5, 5), dtype=np.float32)
    center = grid.shape[0] // 2
    grid[center, center + 1] = 1.0
    info0 = {
        "agent_index": 0,
        "pos": np.array([10.0, 10.0], dtype=np.float32),
        "target_pos": np.array([20.0, 20.0], dtype=np.float32),
    }
    state = _make_state(np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32), info0["target_pos"])
    policy.get_action("drone_0", _make_obs(grid), state, info0)

    assert policy._shared_wall_mask is not None
    assert policy._shared_wall_mask[10, 11] >= 1.0

    empty = np.zeros_like(grid)
    info1 = {
        "agent_index": 1,
        "pos": np.array([10.0, 10.0], dtype=np.float32),
        "target_pos": np.array([20.0, 20.0], dtype=np.float32),
    }
    out = policy._apply_wall_memory(empty, info1, {"wall_memory": policy._shared_wall_memory})
    assert out[center, center + 1] >= 1.0


def test_astar_wall_memory_isolated_when_not_shared():
    policy = AStarGridPolicy(
        memory_shared=False,
        memory_walls_enabled=True,
        memory_threats_enabled=False,
        stuck_steps=0,
        global_plan_enabled=False,
        escape_steps=0,
    )
    grid = np.zeros((5, 5), dtype=np.float32)
    center = grid.shape[0] // 2
    grid[center, center + 1] = 1.0
    info0 = {
        "agent_index": 0,
        "pos": np.array([10.0, 10.0], dtype=np.float32),
        "target_pos": np.array([20.0, 20.0], dtype=np.float32),
    }
    state = _make_state(np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32), info0["target_pos"])
    policy.get_action("drone_0", _make_obs(grid), state, info0)

    empty = np.zeros_like(grid)
    info1 = {
        "agent_index": 1,
        "pos": np.array([10.0, 10.0], dtype=np.float32),
        "target_pos": np.array([20.0, 20.0], dtype=np.float32),
    }
    policy.get_action("drone_1", _make_obs(empty), state, info1)
    state1 = policy._agent_state[1]
    out = policy._apply_wall_memory(empty, info1, state1)
    assert out[center, center + 1] == 0.0


def test_astar_memory_updates_accept_precomputed_base_cell():
    policy = AStarGridPolicy(
        memory_shared=False,
        memory_walls_enabled=True,
        memory_threats_enabled=True,
        memory_threat_persist=1,
        stuck_steps=0,
        global_plan_enabled=False,
        escape_steps=0,
    )
    grid = np.zeros((5, 5), dtype=np.float32)
    center = grid.shape[0] // 2
    grid[center, center + 1] = 1.0
    pos = np.array([10.0, 10.0], dtype=np.float32)
    info = {
        "agent_index": 0,
        "pos": pos,
        "target_pos": np.array([20.0, 20.0], dtype=np.float32),
    }
    state = _make_state(np.array([[10.0, 10.0]], dtype=np.float32), info["target_pos"])
    policy.set_context(state)
    agent_state = policy._agent_state.setdefault(0, policy._default_agent_state())
    agent_state["step"] = 1
    base_cell = policy._cell_from_pos(pos)

    policy._memory_walls_update(agent_state, grid, info, base_cell=base_cell)
    policy._memory_threats_update(agent_state, agent_state, grid, info, base_cell=base_cell)
    out = policy._apply_wall_memory(np.zeros_like(grid), info, agent_state, base_cell=base_cell)

    assert out[center, center + 1] >= 1.0


def test_astar_memory_penalty_grid_matches_with_precomputed_base_cell():
    policy = AStarGridPolicy(
        memory_enabled=True,
        memory_penalty=2.0,
        memory_decay=0.5,
        stuck_steps=0,
        global_plan_enabled=False,
        escape_steps=0,
    )
    agent_state = policy._default_agent_state()
    agent_state["step"] = 5
    agent_state["memory"] = {
        (10, 10): (2.0, 5),
        (11, 10): (4.0, 4),
    }
    info = {"pos": np.array([10.0, 10.0], dtype=np.float32)}
    base_cell = policy._cell_from_pos(info["pos"])

    from_info = policy._memory_penalty_grid((5, 5), agent_state, info=info)
    from_base = policy._memory_penalty_grid((5, 5), agent_state, base_cell=base_cell)

    np.testing.assert_allclose(from_info, from_base)


def test_dijkstra_risk_falls_back_to_python_on_numba_heap_overflow(monkeypatch):
    grid = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 1.0, 2.0],
            [3.0, 2.0, 1.0],
        ],
        dtype=np.float32,
    )
    wall_mask = np.zeros_like(grid, dtype=np.uint8)

    monkeypatch.setattr(astar_mod, "_NUMBA_AVAILABLE", True)

    def fake_numba(grid, start_x, start_y, wall_mask, has_wall_mask, allow_diagonal):
        del start_x, start_y, wall_mask, has_wall_mask, allow_diagonal
        return np.asarray(grid, dtype=np.float32), True

    monkeypatch.setattr(astar_mod, "_dijkstra_risk_numba", fake_numba)

    out = astar_mod._dijkstra_risk(grid, (0, 0), wall_mask, allow_diagonal=False)
    expected = astar_mod._dijkstra_risk_py(grid, (0, 0), wall_mask, allow_diagonal=False)

    np.testing.assert_allclose(out, expected)


def test_astar_falls_back_to_python_on_numba_heap_overflow(monkeypatch):
    grid = np.zeros((4, 4), dtype=np.float32)
    wall_mask = np.zeros_like(grid, dtype=np.uint8)

    monkeypatch.setattr(astar_mod, "_NUMBA_AVAILABLE", True)

    def fake_numba(
        grid,
        start_x,
        start_y,
        goal_x,
        goal_y,
        alpha,
        allow_diagonal,
        penalty_grid,
        has_penalty,
        wall_mask,
        has_wall_mask,
        heuristic_weight,
        turn_penalty,
    ):
        del (
            grid,
            start_x,
            start_y,
            goal_x,
            goal_y,
            alpha,
            allow_diagonal,
            penalty_grid,
            has_penalty,
            wall_mask,
            has_wall_mask,
            heuristic_weight,
            turn_penalty,
        )
        return np.array([0], dtype=np.int32), np.array([0], dtype=np.int32), 1, True

    monkeypatch.setattr(astar_mod, "_astar_numba", fake_numba)

    out = astar_mod._astar(grid, (0, 0), (3, 3), wall_mask=wall_mask)
    expected = astar_mod._astar_py(grid, (0, 0), (3, 3), wall_mask=wall_mask)

    assert out == expected


def test_numba_scratch_is_reused_for_same_shape():
    policy = AStarGridPolicy(numba_warmup=False)

    scratch_a = policy._get_numba_scratch((8, 8))
    scratch_b = policy._get_numba_scratch((8, 8))
    scratch_c = policy._get_numba_scratch((6, 6))

    assert scratch_a is scratch_b
    assert scratch_c is not scratch_a
    assert scratch_c["dist"].shape == (6, 6)


@pytest.mark.skipif(not astar_mod._NUMBA_AVAILABLE, reason="numba не доступна")
def test_astar_and_dijkstra_match_with_scratch_buffers():
    grid = np.array(
        [
            [0.0, 0.1, 0.2, 0.3],
            [0.0, 0.6, 0.7, 0.2],
            [0.0, 0.1, 0.2, 0.1],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    wall_mask = grid >= 0.65
    start = (0, 0)
    goal = (3, 3)
    policy = AStarGridPolicy(numba_warmup=False)
    scratch = policy._get_numba_scratch(grid.shape)

    path_plain = astar_mod._astar(grid, start, goal, wall_mask=wall_mask)
    path_scratch = astar_mod._astar(grid, start, goal, wall_mask=wall_mask, scratch=scratch)
    dist_plain = astar_mod._dijkstra_risk(grid, start, wall_mask, allow_diagonal=True)
    dist_scratch = astar_mod._dijkstra_risk(
        grid,
        start,
        wall_mask,
        allow_diagonal=True,
        dist_buf=policy._get_dijkstra_dist(grid.shape),
        scratch=scratch,
    )

    assert path_scratch == path_plain
    np.testing.assert_allclose(dist_scratch, dist_plain)
