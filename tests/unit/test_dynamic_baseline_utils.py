import numpy as np

from baselines.mpc_lite import MPCLitePolicy
from baselines.utils import velocity_tracking_action


def _make_obs(walls, vel=(0.0, 0.0)):
    vec = np.zeros((13,), dtype=np.float32)
    vec[2] = float(vel[0])
    vec[3] = float(vel[1])
    vec[4:8] = np.asarray(walls, dtype=np.float32)
    return {"vector": vec, "grid": None}


def _make_obs_with_grid(walls, grid, vel=(0.0, 0.0)):
    obs = _make_obs(walls, vel=vel)
    obs["grid"] = np.asarray(grid, dtype=np.float32)
    return obs


def test_velocity_tracking_brakes_near_wall():
    info = {
        "max_speed": 4.0,
        "max_accel": 10.0,
        "dt": 0.1,
        "drag": 0.0,
        "accel_tau": 1.0,
    }
    act_far = velocity_tracking_action(np.array([1.0, 0.0], dtype=np.float32), _make_obs([10, 10, 10, 10]), info)
    act_near = velocity_tracking_action(np.array([1.0, 0.0], dtype=np.float32), _make_obs([10, 0.1, 10, 10]), info)
    assert float(np.linalg.norm(act_near)) < float(np.linalg.norm(act_far))


def test_velocity_tracking_compensates_drag():
    info = {
        "max_speed": 4.0,
        "max_accel": 10.0,
        "dt": 0.1,
        "drag": 0.1,
        "accel_tau": 1.0,
    }
    obs = _make_obs([10, 10, 10, 10], vel=(0.5, 0.0))
    act = velocity_tracking_action(np.array([0.5, 0.0], dtype=np.float32), obs, info)
    assert act[0] > 0.0


def test_velocity_tracking_brakes_on_grid_wall():
    grid = np.zeros((5, 5), dtype=np.float32)
    grid[2, 3] = 1.0
    info = {
        "max_speed": 4.0,
        "max_accel": 1.0,
        "dt": 0.1,
        "drag": 0.0,
        "grid_res": 1.0,
    }
    obs_clear = _make_obs_with_grid([10, 10, 10, 10], np.zeros_like(grid), vel=(1.0, 0.0))
    obs_wall = _make_obs_with_grid([10, 10, 10, 10], grid, vel=(1.0, 0.0))
    act_clear = velocity_tracking_action(np.array([1.0, 0.0], dtype=np.float32), obs_clear, info)
    act_wall = velocity_tracking_action(np.array([1.0, 0.0], dtype=np.float32), obs_wall, info)
    assert act_wall[0] < act_clear[0]


def test_mpc_jerk_penalty_applies():
    policy = MPCLitePolicy(jerk_penalty=1.0, fallback_astar=False)
    state = policy._agent_state.setdefault(0, {"best_dist": None, "since_improve": 0, "last_action": None})
    state["last_action"] = np.array([0.0, 0.0], dtype=np.float32)
    actions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    scores = np.array([0.0, 0.0], dtype=np.float32)
    penalized = policy._apply_action_penalties(0, actions, scores)
    assert penalized[1] < penalized[0]


def test_mpc_terminal_speed_penalty():
    policy = MPCLitePolicy(terminal_speed_weight=1.0, terminal_brake_radius=2.0, fallback_astar=False)
    scores = np.zeros((2,), dtype=np.float32)
    pos_end = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    vel_end = np.array([[1.0, 0.0], [0.1, 0.0]], dtype=np.float32)
    target = np.array([0.5, 0.0], dtype=np.float32)
    out = policy._apply_terminal_speed_penalty(scores, pos_end, vel_end, target, max_speed=1.0)
    assert out[0] < out[1]
