from __future__ import annotations

import numpy as np

from baselines.astar_support import (
    MOVES_CARDINAL,
    MOVES_DIAGONAL,
    dist_from_info as astar_dist_from_info,
    escape_direction,
    memory_penalty_from_entries,
    reuse_direction_to_cell,
)
from baselines.mpc_support import (
    dist_from_info as mpc_dist_from_info,
    extract_plan_action,
    finalize_batch_waypoint_actions,
    near_agents_penalty,
)


def test_astar_support_exports_expected_moves():
    assert len(MOVES_CARDINAL) == 4
    assert len(MOVES_DIAGONAL) == 8


def test_astar_escape_direction_prefers_low_cost_neighbor():
    grid = np.ones((5, 5), dtype=np.float32)
    grid[2, 3] = 0.0

    out = escape_direction(
        grid,
        np.array([1.0, 0.0], dtype=np.float32),
        allow_diagonal=False,
        oracle_dir=np.array([1.0, 0.0], dtype=np.float32),
    )

    assert out is not None
    assert np.allclose(out, np.array([1.0, 0.0], dtype=np.float32))


def test_astar_and_mpc_dist_helpers_use_info_first():
    info = {"dist": 3.5, "pos": np.array([0.0, 0.0], dtype=np.float32), "target_pos": np.array([1.0, 1.0])}
    astar_dist, astar_normed = astar_dist_from_info(info, np.array([1.0, 1.0], dtype=np.float32))
    mpc_dist = mpc_dist_from_info(info, {"vector": np.zeros((2,), dtype=np.float32), "grid": None})

    assert astar_dist == 3.5
    assert astar_normed is False
    assert mpc_dist == 3.5


def test_mpc_extract_plan_action_and_neighbor_penalty():
    action = extract_plan_action((np.array([0.25, -0.5], dtype=np.float32), {"score": 1.0}))
    pos = np.array([[0.0, 0.0], [0.5, 0.0], [5.0, 5.0]], dtype=np.float32)
    alive = np.array([True, True, False])

    assert np.allclose(action, np.array([0.25, -0.5], dtype=np.float32))
    assert near_agents_penalty(0, pos, alive) > 0.0


def test_astar_support_memory_penalty_and_reuse_vector():
    penalty, stale = memory_penalty_from_entries(
        np.array([[10, 10], [11, 10]], dtype=np.int32),
        np.array([2.0, 4.0], dtype=np.float32),
        np.array([5, 4], dtype=np.int32),
        base_cell=(10, 10),
        step=5,
        shape=(5, 5),
        memory_penalty=2.0,
        memory_decay=0.5,
    )
    assert penalty is not None
    assert stale.shape == (0, 2)
    vec = reuse_direction_to_cell((11, 10), np.array([10.0, 10.0], dtype=np.float32), cell_size=1.0)
    assert vec.shape == (2,)
    assert np.linalg.norm(vec) > 0.0


def test_mpc_finalize_batch_waypoint_actions_without_controller():
    out = finalize_batch_waypoint_actions(
        agent_ids=["a0", "a1"],
        desired=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        to_target=np.zeros((2, 2), dtype=np.float32),
        dist_m=np.zeros((2,), dtype=np.float32),
        in_goal=np.zeros((2,), dtype=bool),
        risk_p=np.zeros((2,), dtype=np.float32),
        cur_vel=np.zeros((2, 2), dtype=np.float32),
        obs_list=[{"vector": np.zeros((13,), dtype=np.float32), "grid": None}] * 2,
        info_list=[{}, {}],
        state=type("State", (), {"field_size": 100.0, "max_speed": 5.0, "max_accel": 2.0, "dt": 0.1, "drag": 0.0})(),
        controller=None,
        stop_risk_threshold=0.4,
    )
    assert set(out) == {"a0", "a1"}
    assert np.allclose(out["a0"], np.array([1.0, 0.0], dtype=np.float32))
