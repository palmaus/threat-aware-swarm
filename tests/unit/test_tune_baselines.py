import json
import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.tuning.tune_baselines as tb
from baselines.astar_grid import AStarGridPolicy
from env.config import EnvConfig
from env.pz_env import SwarmPZEnv
from env.state import SimState


def _stub_metrics():
    return {
        "success": True,
        "finished_frac_end": 1.0,
        "alive_frac_end": 1.0,
        "deaths": 0,
        "time_to_goal_mean": 10.0,
        "risk_integral_all": 0.1,
        "risk_integral_alive": 0.1,
        "collision_like": 0.0,
        "episode_len": 5,
    }


def test_score_scalar_constraints():
    agg = {
        "finished_frac_end": 0.5,
        "alive_frac_end": 1.0,
        "deaths_mean": 0.0,
        "risk_integral_all": 0.1,
        "time_to_goal_mean": 10.0,
    }
    bad_finish = tb._score_scalar(agg, finish_min=0.95, alive_min=0.95)
    assert bad_finish < -1.0e8

    agg["finished_frac_end"] = 0.98
    agg["alive_frac_end"] = 0.5
    bad_alive = tb._score_scalar(agg, finish_min=0.95, alive_min=0.95)
    assert bad_alive < -1.0e7

    agg["alive_frac_end"] = 0.98
    ok = tb._score_scalar(agg, finish_min=0.95, alive_min=0.95)
    assert ok > 0.0


def test_evaluate_policy_crn_seeds(monkeypatch):
    seeds_used = []

    def fake_run_episode(env, policy, scene, seed, success_threshold):
        seeds_used.append((scene["id"], seed))
        return _stub_metrics()

    monkeypatch.setattr(tb, "run_episode", fake_run_episode)

    env = SwarmPZEnv(EnvConfig(), max_steps=10, goal_radius=3.0)
    policy = AStarGridPolicy()
    scenes = [{"id": "A"}, {"id": "B"}]
    eval_seeds = [10, 11]

    tb.evaluate_policy(
        env,
        policy,
        scenes,
        eval_seeds,
        success_threshold=0.5,
        finish_min=0.95,
        alive_min=0.95,
    )

    expected = []
    for scene in scenes:
        offset = tb._stable_hash(scene["id"]) % 100000
        for eval_seed in eval_seeds:
            expected.append((scene["id"], eval_seed + offset))
    assert seeds_used == expected

    seeds_used.clear()
    tb.evaluate_policy(
        env,
        policy,
        scenes,
        eval_seeds,
        success_threshold=0.5,
        finish_min=0.95,
        alive_min=0.95,
    )
    assert seeds_used == expected


def test_evaluate_policy_prune_after_min_scenes(monkeypatch):
    try:
        import optuna
    except Exception:  # pragma: no cover - optuna является зависимостью для tune_baselines.
        pytest.skip("optuna not available")

    class DummyTrial:
        def __init__(self):
            self.reported = []
            self.user_attrs = {}

        def report(self, value, step):
            self.reported.append((step, value))

        def should_prune(self):
            return True

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    def fake_run_episode(env, policy, scene, seed, success_threshold):
        return _stub_metrics()

    monkeypatch.setattr(tb, "run_episode", fake_run_episode)

    env = SwarmPZEnv(EnvConfig(), max_steps=10, goal_radius=3.0)
    policy = AStarGridPolicy()
    scenes = [{"id": "S1"}, {"id": "S2"}]
    eval_seeds = [0, 1]
    trial = DummyTrial()

    with pytest.raises(optuna.TrialPruned):
        tb.evaluate_policy(
            env,
            policy,
            scenes,
            eval_seeds,
            success_threshold=0.5,
            finish_min=0.95,
            alive_min=0.95,
            trial=trial,
            min_prune_scenes=2,
        )

    assert trial.user_attrs.get("reported_step") == 2
    assert "aggregate_partial" in trial.user_attrs


def test_evaluate_policy_reuses_eval_cache(monkeypatch, tmp_path):
    calls = []

    def fake_run_episode(env, policy, scene, seed, success_threshold):
        calls.append((scene["id"], seed))
        return _stub_metrics()

    monkeypatch.setattr(tb, "run_episode", fake_run_episode)

    env = SwarmPZEnv(EnvConfig(), max_steps=10, goal_radius=3.0)
    policy = AStarGridPolicy()
    scenes = [{"id": "cache_scene"}]
    eval_seeds = [3]
    cache_args = SimpleNamespace(eval_cache=True, cache_dir=str(tmp_path), out_root="runs", success_threshold=0.5)
    eval_cache = tb._make_eval_cache(
        cache_args,
        out_dir=tmp_path,
        env_payload={"field_size": 100.0},
        regime="fair",
        goal_radius=3.0,
    )

    tb.evaluate_policy(
        env,
        policy,
        scenes,
        eval_seeds,
        success_threshold=0.5,
        finish_min=0.95,
        alive_min=0.95,
        eval_cache=eval_cache,
        cache_policy_name="baseline:astar_grid",
        cache_params={"alpha": 1.0},
    )
    tb.evaluate_policy(
        env,
        policy,
        scenes,
        eval_seeds,
        success_threshold=0.5,
        finish_min=0.95,
        alive_min=0.95,
        eval_cache=eval_cache,
        cache_policy_name="baseline:astar_grid",
        cache_params={"alpha": 1.0},
    )

    assert len(calls) == 1
    assert eval_cache["stats"]["hits"] == 1
    assert eval_cache["stats"]["writes"] == 1


def test_make_eval_cache_uses_versioned_namespace(tmp_path):
    args = SimpleNamespace(
        eval_cache=True,
        cache_dir=str(tmp_path),
        out_root="runs",
        success_threshold=0.5,
        cache_version="v-test",
        cache_cleanup=False,
        cache_ttl_days=14,
    )
    cache = tb._make_eval_cache(
        args,
        out_dir=tmp_path,
        env_payload={"field_size": 100.0},
        regime="fair",
        goal_radius=3.0,
    )
    assert cache["namespace"].startswith("v-test_")
    assert cache["root"].parent == tmp_path
    assert cache["code_hash"]


def test_make_eval_cache_cleans_stale_namespaces(tmp_path):
    stale = tmp_path / "v0_oldhash"
    stale.mkdir(parents=True)
    stale_file = stale / "entry.json"
    stale_file.write_text("{}", encoding="utf-8")
    old_ts = time.time() - (20 * 24 * 60 * 60)
    os.utime(stale_file, (old_ts, old_ts))
    os.utime(stale, (old_ts, old_ts))

    args = SimpleNamespace(
        eval_cache=True,
        cache_dir=str(tmp_path),
        out_root="runs",
        success_threshold=0.5,
        cache_version="v-clean",
        cache_cleanup=True,
        cache_ttl_days=7,
    )
    cache = tb._make_eval_cache(
        args,
        out_dir=tmp_path,
        env_payload={"field_size": 100.0},
        regime="fair",
        goal_radius=3.0,
    )
    assert not stale.exists()
    assert cache["cleanup"]["removed_namespaces"] >= 1


def test_merge_best_params(tmp_path):
    out_path = tmp_path / "best_policy_params.json"
    out_path.write_text(
        json.dumps(
            {
                "policies": {"baseline:existing": {"alpha": 1.0}},
                "meta": {"source": "old"},
            }
        ),
        encoding="utf-8",
    )

    stageb_best = {
        "baseline:astar_grid": {"params": {"alpha": 3.5, "max_cost": 0.3}},
    }
    tb._merge_best_params(stageb_best, out_path, meta={"source": "stageB"})

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["policies"]["baseline:existing"]["alpha"] == 1.0
    assert data["policies"]["baseline:astar_grid"]["alpha"] == 3.5
    assert data["meta"]["source"] == "stageB"


def test_summarize_results_tracks_validation_status():
    summary = tb._summarize_results(
        {
            "baseline:astar_grid": {
                "trials": [{"state": "COMPLETE"}],
                "best": {"score_scalar": 1.0},
                "validation_status": "search_only",
                "promotion_status": "not_promoted",
            }
        }
    )
    assert summary["baseline:astar_grid"]["states"]["COMPLETE"] == 1
    assert summary["baseline:astar_grid"]["validation_status"] == "search_only"
    assert summary["baseline:astar_grid"]["promotion_status"] == "not_promoted"


def test_astar_near_goal_speed_cap():
    policy = AStarGridPolicy(
        goal_radius_control=5.0,
        near_goal_speed_cap=0.2,
        near_goal_kp=1.0,
        near_goal_damping=0.0,
        risk_speed_scale=0.0,
        stop_risk_threshold=0.0,
    )
    vec = np.zeros((13,), dtype=np.float32)
    vec[0] = 0.1
    obs = {
        "vector": vec,
        "grid": np.zeros((1, 41, 41), dtype=np.float32),
    }
    target = np.array([1.0, 0.0], dtype=np.float32)
    pos = np.array([[0.0, 0.0]], dtype=np.float32)
    state = SimState(
        pos=pos,
        vel=np.zeros_like(pos),
        alive=np.array([True]),
        target_pos=target,
        target_vel=np.zeros((2,), dtype=np.float32),
        timestep=0,
        threats=[],
        dists=np.array([1.0], dtype=np.float32),
        in_goal=np.array([False]),
        in_goal_steps=np.array([0], dtype=np.int32),
        finished=np.array([False]),
        newly_finished=np.array([False]),
        risk_p=np.array([0.5], dtype=np.float32),
        min_neighbor_dist=np.array([np.inf], dtype=np.float32),
        last_action=np.zeros((1, 2), dtype=np.float32),
        walls=np.zeros((1, 4), dtype=np.float32),
        oracle_dir=None,
        static_walls=[],
        static_circles=[],
        collision_speed=np.zeros((1,), dtype=np.float32),
        measured_accel=np.zeros_like(pos, dtype=np.float32),
        energy=np.array([100.0], dtype=np.float32),
        energy_level=np.array([1.0], dtype=np.float32),
        agent_state=np.array([0], dtype=np.int8),
        field_size=10.0,
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
    info = {
        "dist": 1.0,
        "in_goal": 0.0,
        "risk_p": 0.5,
        "agent_index": 0,
        "pos": pos[0],
        "target_pos": target,
    }
    action = policy.get_action("drone_0", obs, state, info)
    assert np.linalg.norm(action) <= 0.21


def test_open_trial_csv_append(tmp_path):
    fieldnames = ["policy", "success_rate"]
    csv_path = tmp_path / "tune.csv"
    f, writer = tb._open_trial_csv(csv_path, fieldnames)
    writer.writerow({"policy": "a", "success_rate": 1.0})
    f.close()

    f2, writer2 = tb._open_trial_csv(csv_path, fieldnames, append=True)
    writer2.writerow({"policy": "b", "success_rate": 0.5})
    f2.close()

    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].startswith("policy")
    assert len(lines) == 3


def test_build_tuning_protocol_derives_splits_and_seed_packs(tmp_path):
    scene_paths = []
    scenes = []
    payloads = [
        {"id": "S0_sanity_no_threats", "threats": [], "max_steps": 10},
        {"id": "S1_single_threat_blocking", "threats": [{"x": 1}], "max_steps": 10},
        {"id": "S8_u_trap_escape", "walls": [{"x": 1}], "threats": [{"x": 1}], "max_steps": 10},
        {"id": "S18_hard_maze", "walls": [{"x": 1}], "threats": [{"x": 1}], "max_steps": 10},
    ]
    for idx, payload in enumerate(payloads):
        path = tmp_path / f"s{idx}.yaml"
        path.write_text(json.dumps(payload), encoding="utf-8")
        scene_paths.append(path)
        scenes.append(payload)

    args = SimpleNamespace(
        scenes=[],
        search_scenes=[],
        holdout_scenes=[],
        benchmark_scenes=[],
        ood_scenes=[],
        seed=7,
        episodes=2,
        episodes_eval=3,
        stage_b=True,
        search_seeds=[],
        holdout_seeds=[],
        report_seeds=[],
    )

    protocol = tb._build_tuning_protocol(args, scene_paths, scenes)

    assert protocol["search_scenes"]
    assert protocol["holdout_scenes"]
    assert protocol["search_seeds"] == [7, 8]
    assert protocol["holdout_seeds"] == [10007, 10008, 10009]
    assert protocol["report_seeds"] == [20007, 20008, 20009]
    assert "search_scene_hash" in protocol["summary"]


def test_build_tuning_protocol_respects_explicit_splits(tmp_path, monkeypatch):
    search_path = tmp_path / "search.yaml"
    holdout_path = tmp_path / "holdout.yaml"
    benchmark_path = tmp_path / "benchmark.yaml"
    for path, scene_id in [
        (search_path, "S0_sanity_no_threats"),
        (holdout_path, "S17_shared_memory_gap"),
        (benchmark_path, "S18_hard_maze"),
    ]:
        path.write_text(json.dumps({"id": scene_id, "max_steps": 10}), encoding="utf-8")

    def fake_resolve(paths):
        mapping = {
            "search": [search_path],
            "holdout": [holdout_path],
            "benchmark": [benchmark_path],
            "preset:ood": [],
        }
        if not paths:
            return []
        out = []
        for item in paths:
            out.extend(mapping[item])
        return out

    monkeypatch.setattr(tb, "resolve_scene_paths", fake_resolve)
    monkeypatch.setattr(tb, "load_scenes", lambda paths: [json.loads(path.read_text(encoding="utf-8")) for path in paths])

    args = SimpleNamespace(
        scenes=[],
        search_scenes=["search"],
        holdout_scenes=["holdout"],
        benchmark_scenes=["benchmark"],
        ood_scenes=[],
        seed=1,
        episodes=1,
        episodes_eval=1,
        stage_b=True,
        search_seeds=[],
        holdout_seeds=[],
        report_seeds=[],
    )

    protocol = tb._build_tuning_protocol(args, [], [])
    assert [scene["id"] for scene in protocol["search_scenes"]] == ["S0_sanity_no_threats"]
    assert [scene["id"] for scene in protocol["holdout_scenes"]] == ["S17_shared_memory_gap"]
    assert [scene["id"] for scene in protocol["benchmark_scenes"]] == ["S18_hard_maze"]


def test_aggregate_by_family_uses_macro_average():
    per_scene = {
        "easy_a": {"finished_frac_end": 1.0, "alive_frac_end": 1.0, "deaths_mean": 0.0, "risk_integral_all": 0.0, "risk_integral_alive": 0.0, "time_to_goal_mean": 10.0, "collision_like": 0.0, "episode_len_mean": 10.0, "path_ratio": 1.0, "action_smoothness": 0.1, "energy_efficiency": 1.0, "safety_score": 1.0, "cost_progress": 0.0, "cost_risk": 0.0, "cost_wall": 0.0, "cost_collision": 0.0, "cost_energy": 0.0, "cost_jerk": 0.0, "cost_time": 0.0, "success_rate": 1.0},
        "easy_b": {"finished_frac_end": 1.0, "alive_frac_end": 1.0, "deaths_mean": 0.0, "risk_integral_all": 0.0, "risk_integral_alive": 0.0, "time_to_goal_mean": 10.0, "collision_like": 0.0, "episode_len_mean": 10.0, "path_ratio": 1.0, "action_smoothness": 0.1, "energy_efficiency": 1.0, "safety_score": 1.0, "cost_progress": 0.0, "cost_risk": 0.0, "cost_wall": 0.0, "cost_collision": 0.0, "cost_energy": 0.0, "cost_jerk": 0.0, "cost_time": 0.0, "success_rate": 1.0},
        "hard_a": {"finished_frac_end": 0.0, "alive_frac_end": 1.0, "deaths_mean": 0.0, "risk_integral_all": 0.0, "risk_integral_alive": 0.0, "time_to_goal_mean": 100.0, "collision_like": 0.0, "episode_len_mean": 100.0, "path_ratio": 2.0, "action_smoothness": 0.1, "energy_efficiency": 1.0, "safety_score": 1.0, "cost_progress": 0.0, "cost_risk": 0.0, "cost_wall": 0.0, "cost_collision": 0.0, "cost_energy": 0.0, "cost_jerk": 0.0, "cost_time": 0.0, "success_rate": 0.0},
    }
    family_map = {"easy_a": "easy", "easy_b": "easy", "hard_a": "hard"}
    agg = tb._aggregate_by_family(per_scene, family_map)
    assert agg["finished_frac_end"] == pytest.approx(0.5)


def test_select_champions_and_pareto():
    args = SimpleNamespace(
        finish_min=0.7,
        alive_min=0.7,
        gate_finish_min=-1.0,
        gate_alive_min=-1.0,
        gate_max_risk=-1.0,
    )
    rows = [
        {"name": "balanced", "value": 100.0, "aggregate": {"finished_frac_end": 0.9, "alive_frac_end": 0.95, "risk_integral_alive": 0.05, "time_to_goal_mean": 40.0, "finished_frac_end_std": 0.10}},
        {"name": "safe", "value": 80.0, "aggregate": {"finished_frac_end": 0.85, "alive_frac_end": 1.0, "risk_integral_alive": 0.01, "time_to_goal_mean": 60.0, "finished_frac_end_std": 0.01}},
        {"name": "fast", "value": 70.0, "aggregate": {"finished_frac_end": 0.8, "alive_frac_end": 0.9, "risk_integral_alive": 0.03, "time_to_goal_mean": 20.0, "finished_frac_end_std": 0.03}},
        {"name": "bad", "value": 120.0, "aggregate": {"finished_frac_end": 0.1, "alive_frac_end": 0.2, "risk_integral_alive": 0.9, "time_to_goal_mean": 5.0, "finished_frac_end_std": 0.30}},
    ]
    champions = tb._select_champions(rows, args=args)
    frontier = tb._pareto_frontier(rows, args=args)
    assert champions["balanced"]["name"] == "balanced"
    assert champions["safe"]["name"] == "safe"
    assert champions["fast"]["name"] == "fast"
    assert champions["stable"]["name"] == "safe"
    assert all(row["name"] != "bad" for row in frontier)


def test_validate_information_regime_blocks_oracle_for_fair():
    args = SimpleNamespace(information_regime="fair")
    cfg = EnvConfig()
    cfg.oracle_visibility = "baseline"
    with pytest.raises(SystemExit):
        tb._validate_information_regime(args, cfg)


def test_validate_policy_params_for_regime_blocks_oracle_like_params():
    with pytest.raises(SystemExit):
        tb._validate_policy_params_for_regime(
            "baseline:astar_grid",
            {"frontier_oracle_weight": 0.5},
            "fair",
        )


def test_optuna_namespaces_include_regime(tmp_path):
    db_path = tb._resolve_optuna_storage_path(
        tmp_path,
        "baseline:astar_grid",
        "runs/test.db",
        multi_policy=True,
        regime="fair",
    )
    assert db_path.name.endswith("_fair_baseline_astar_grid.db")

    args = SimpleNamespace(study_name="tune_static", policy=["all"])
    study_name = tb._resolve_study_name(
        args,
        "baseline:astar_grid",
        storage_url="sqlite:///unused.db",
        resume_mode=False,
        regime="fair",
    )
    assert study_name.endswith("_fair")


def test_resolve_search_stages_progressively_reduces_budget():
    args = SimpleNamespace(tuning_profile="fast", search_fidelity=[])
    scenes = [
        {"id": "S0_sanity_no_threats"},
        {"id": "S8_u_trap_escape", "walls": [{"x": 1}]},
        {"id": "S11_dynamic_gate", "dynamic_threats": [{"x": 1}]},
        {"id": "S18_hard_maze", "walls": [{"x": 1}]},
    ]
    families = tb._scene_family_map(scenes)
    stages = tb._resolve_search_stages(
        args,
        search_scenes=scenes,
        eval_seeds=[1, 2, 3],
        max_steps=100,
        scene_families=families,
    )
    assert len(stages) >= 2
    assert stages[0]["max_steps"] < stages[-1]["max_steps"]
    assert len(stages[0]["seeds"]) <= len(stages[-1]["seeds"])
    assert len(stages[0]["scenes"]) <= len(stages[-1]["scenes"])


def test_persistent_runtime_disabled_for_parallel_optuna() -> None:
    assert tb._persistent_runtime_enabled("optuna", 8, True) is False
    assert tb._persistent_runtime_enabled("optuna", 1, True) is True
    assert tb._persistent_runtime_enabled("random", 8, True) is True
    assert tb._persistent_runtime_enabled("optuna", 8, False) is False
    assert tb._persistent_runtime_enabled("optuna", 8, True, optuna_parallel_backend="process") is True


def test_persistent_runtime_can_be_forced_in_parallel_debug() -> None:
    assert (
        tb._persistent_runtime_enabled(
            "optuna",
            8,
            True,
            allow_unsafe_persistent_runtime=True,
        )
        is True
    )


def test_effective_optuna_n_jobs_forces_serial_execution() -> None:
    assert tb._effective_optuna_n_jobs("optuna", 8) == 1
    assert tb._effective_optuna_n_jobs("optuna", 1) == 1
    assert tb._effective_optuna_n_jobs("random", 8) == 8


def test_effective_optuna_n_jobs_allows_unsafe_debug_override() -> None:
    assert tb._effective_optuna_n_jobs("optuna", 8, allow_unsafe_parallel_optuna=True) == 8


def test_resolve_optuna_parallel_backend() -> None:
    assert tb._resolve_optuna_parallel_backend("optuna", 1) == "serial"
    assert tb._resolve_optuna_parallel_backend("optuna", 4) == "process"
    assert tb._resolve_optuna_parallel_backend("optuna", 4, allow_unsafe_parallel_optuna=True) == "thread_debug"
    assert tb._resolve_optuna_parallel_backend("random", 4) == "serial"


def test_resolve_process_worker_count_caps_by_memory(monkeypatch) -> None:
    monkeypatch.setattr(
        tb,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=8 * 1024**3)),
    )
    args = SimpleNamespace(
        n_jobs=4,
        parallel_max_workers=0,
        parallel_reserve_cpus=2,
        parallel_memory_fraction=0.7,
        tuning_profile="balanced",
        step_budget_name="static",
    )

    effective, meta = tb._resolve_process_worker_count(args, "baseline:astar_grid")

    assert effective == 3
    assert meta["requested"] == 4
    assert meta["mem_cap"] == 3


def test_resolve_process_worker_count_respects_explicit_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        tb,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=64 * 1024**3)),
    )
    args = SimpleNamespace(
        n_jobs=8,
        parallel_max_workers=3,
        parallel_reserve_cpus=1,
        parallel_memory_fraction=0.7,
        tuning_profile="fast",
        step_budget_name="default",
    )

    effective, meta = tb._resolve_process_worker_count(args, "baseline:potential_fields")

    assert effective == 3
    assert meta["cpu_cap"] == 3


def test_resolve_process_worker_count_applies_policy_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        tb,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=64 * 1024**3)),
    )
    args = SimpleNamespace(
        n_jobs=8,
        parallel_max_workers=0,
        parallel_reserve_cpus=1,
        parallel_memory_fraction=0.7,
        tuning_profile="balanced",
        step_budget_name="static",
    )

    effective, meta = tb._resolve_process_worker_count(args, "baseline:astar_grid")

    assert effective == 8
    assert meta["policy_cap"] == 8


def test_resolve_policy_cache_size_applies_policy_cap() -> None:
    args = SimpleNamespace(policy_cache_size=8)
    assert tb._resolve_policy_cache_size(args, "baseline:astar_grid") == 1
    assert tb._resolve_policy_cache_size(args, "baseline:mpc_lite") == 2
    assert tb._resolve_policy_cache_size(args, "baseline:flow_field") == 4


def test_resolve_process_worker_trial_cap_applies_policy_default() -> None:
    args = SimpleNamespace(
        parallel_max_trials_per_worker=0,
        tuning_profile="balanced",
        step_budget_name="static",
    )

    effective, meta = tb._resolve_process_worker_trial_cap(args, "baseline:astar_grid")

    assert effective == 1
    assert meta["default_cap"] == 1
    assert meta["requested"] == 0


def test_resolve_process_worker_trial_cap_respects_safe_upper_bound() -> None:
    args = SimpleNamespace(
        parallel_max_trials_per_worker=8,
        tuning_profile="deep",
        step_budget_name="long",
    )

    effective, meta = tb._resolve_process_worker_trial_cap(args, "baseline:mpc_lite")

    assert effective == 1
    assert meta["default_cap"] == 1
    assert meta["requested"] == 8


def test_build_process_worker_batches_recycles_workers() -> None:
    batches = tb._build_process_worker_batches(total_trials=7, max_workers=3, trial_cap=2)

    assert batches == [
        [(0, 2), (1, 2), (2, 2)],
        [(3, 1)],
    ]


def test_optuna_process_worker_initializer_sets_guards(monkeypatch) -> None:
    called = {"env": 0, "deathsig": 0, "warmup": 0}

    monkeypatch.setattr(tb, "_apply_worker_env_vars", lambda: called.__setitem__("env", called["env"] + 1))
    monkeypatch.setattr(tb, "_set_parent_death_signal", lambda sig=tb.signal.SIGTERM: called.__setitem__("deathsig", called["deathsig"] + 1) or True)
    monkeypatch.setattr(tb, "_warmup_worker_numba", lambda: called.__setitem__("warmup", called["warmup"] + 1))
    monkeypatch.setattr(tb.os, "getppid", lambda: 4242)

    tb._optuna_process_worker_initializer(4242)

    assert called == {"env": 1, "deathsig": 1, "warmup": 1}


def test_optuna_process_worker_initializer_exits_for_orphan(monkeypatch) -> None:
    monkeypatch.setattr(tb, "_apply_worker_env_vars", lambda: None)
    monkeypatch.setattr(tb, "_set_parent_death_signal", lambda sig=tb.signal.SIGTERM: True)
    monkeypatch.setattr(tb, "_warmup_worker_numba", lambda: None)
    monkeypatch.setattr(tb.os, "getppid", lambda: 1)

    with pytest.raises(SystemExit):
        tb._optuna_process_worker_initializer(4242)


def test_terminate_spawn_children_kills_only_spawn_workers(monkeypatch) -> None:
    events = []

    class DummyChild:
        def __init__(self, pid, cmdline):
            self.pid = pid
            self._cmdline = cmdline

        def cmdline(self):
            return self._cmdline

        def send_signal(self, sig):
            events.append(("signal", self.pid, sig))

        def kill(self):
            events.append(("kill", self.pid))

    children = [
        DummyChild(10, ["python3", "-c", "from multiprocessing.spawn import spawn_main"]),
        DummyChild(11, ["python3", "other_script.py"]),
    ]

    class DummyProc:
        def children(self, recursive=True):
            assert recursive is True
            return children

    monkeypatch.setattr(tb, "psutil", SimpleNamespace(Process=lambda _pid: DummyProc(), wait_procs=lambda procs, timeout: ([], procs[:1])))
    monkeypatch.setattr(tb.os, "getpid", lambda: 999)

    tb._terminate_spawn_children(wait_s=0.01)

    assert ("signal", 10, tb.signal.SIGTERM) in events
    assert ("kill", 10) in events
    assert all(event[1] != 11 for event in events)


def test_resolve_search_stages_uses_policy_override_for_fast_astar():
    args = SimpleNamespace(tuning_profile="fast", search_fidelity=[], policy_search_fidelity={})
    scenes = [
        {"id": "S0_sanity_no_threats"},
        {"id": "S18_hard_maze", "walls": [{"x": 1}]},
    ]
    families = tb._scene_family_map(scenes)
    default_stages = tb._resolve_search_stages(
        args,
        search_scenes=scenes,
        eval_seeds=[1],
        max_steps=100,
        scene_families=families,
    )
    astar_stages = tb._resolve_search_stages(
        args,
        search_scenes=scenes,
        eval_seeds=[1],
        max_steps=100,
        scene_families=families,
        policy_name="baseline:astar_grid",
    )
    assert len(default_stages) == 3
    assert len(astar_stages) == 2


def test_evaluate_candidate_progressive_tracks_stage_results(monkeypatch):
    captured_max_steps = []

    def fake_make_env(env_payload, *, max_steps, goal_radius, lite_metrics):
        captured_max_steps.append(max_steps)
        return SimpleNamespace(max_steps=max_steps)

    def fake_create_policy(policy_name, params, env, seed):
        return {"policy_name": policy_name, "params": params, "seed": seed}

    def fake_evaluate_policy(
        env,
        policy,
        scenes,
        eval_seeds,
        success_threshold,
        finish_min,
        alive_min,
        trial=None,
        min_prune_scenes=0,
        prune_score_fn=None,
        aggregate_fn=None,
        eval_cache=None,
        cache_policy_name=None,
        cache_params=None,
        **kwargs,
    ):
        per_scene = {}
        for _idx, scene in enumerate(scenes):
            per_scene[scene["id"]] = {
                "success_rate": 1.0,
                "finished_frac_end": min(1.0, 0.5 + 0.1 * len(eval_seeds)),
                "alive_frac_end": 1.0,
                "deaths_mean": 0.0,
                "time_to_goal_mean": float(env.max_steps),
                "risk_integral_all": 0.0,
                "risk_integral_alive": 0.0,
                "collision_like": 0.0,
                "episode_len_mean": float(env.max_steps),
                "path_ratio": 1.0,
                "action_smoothness": 0.1,
                "energy_efficiency": 1.0,
                "safety_score": 1.0,
                "cost_progress": 0.0,
                "cost_risk": 0.0,
                "cost_wall": 0.0,
                "cost_collision": 0.0,
                "cost_energy": 0.0,
                "cost_jerk": 0.0,
                "cost_time": 0.0,
            }
        return per_scene

    monkeypatch.setattr(tb, "_make_env", fake_make_env)
    monkeypatch.setattr(tb, "create_policy", fake_create_policy)
    monkeypatch.setattr(tb, "evaluate_policy", fake_evaluate_policy)

    stages = [
        {
            "label": "warmup",
            "scenes": [{"id": "S0_sanity_no_threats"}],
            "scene_ids": ["S0_sanity_no_threats"],
            "scene_families": {"S0_sanity_no_threats": "static_open_no_threat"},
            "seeds": [1],
            "max_steps": 20,
        },
        {
            "label": "full",
            "scenes": [{"id": "S0_sanity_no_threats"}, {"id": "S18_hard_maze"}],
            "scene_ids": ["S0_sanity_no_threats", "S18_hard_maze"],
            "scene_families": {
                "S0_sanity_no_threats": "static_open_no_threat",
                "S18_hard_maze": "static_maze_navigation",
            },
            "seeds": [1, 2],
            "max_steps": 80,
        },
    ]

    trial = tb._evaluate_candidate_progressive(
        policy_name="baseline:astar_grid",
        params={"alpha": 1.0},
        stages=stages,
        success_threshold=0.5,
        finish_min=0.7,
        alive_min=0.7,
        save_scenes=False,
        lite_metrics=True,
        env_payload={},
        goal_radius=3.0,
        aggregate_by_family=True,
        base_seed=0,
    )

    assert captured_max_steps == [20, 80]
    assert len(trial["search_stages"]) == 2
    assert trial["search_stages"][-1]["label"] == "full"


def test_evaluate_candidate_progressive_reuses_persistent_env_and_policy(monkeypatch):
    env_calls = []
    policy_calls = []

    class DummyEnv(SimpleNamespace):
        pass

    def fake_make_env(env_payload, *, max_steps, goal_radius, lite_metrics):
        env_calls.append(max_steps)
        return DummyEnv(
            max_steps=max_steps,
            goal_radius=goal_radius,
            config=SimpleNamespace(
                field_size=100.0,
                n_agents=20,
                grid_width=41,
                grid_res=1.0,
                oracle_visibility="none",
                debug_metrics_mode="lite" if lite_metrics else "full",
            ),
        )

    def fake_create_policy(policy_name, params, env, seed):
        policy_calls.append((policy_name, env.max_steps, seed))
        return {"policy_name": policy_name, "max_steps": env.max_steps}

    def fake_evaluate_policy(
        env,
        policy,
        scenes,
        eval_seeds,
        success_threshold,
        finish_min,
        alive_min,
        trial=None,
        min_prune_scenes=0,
        prune_score_fn=None,
        aggregate_fn=None,
        eval_cache=None,
        cache_policy_name=None,
        cache_params=None,
        **kwargs,
    ):
        return {
            scene["id"]: {
                "success_rate": 1.0,
                "finished_frac_end": 1.0,
                "alive_frac_end": 1.0,
                "deaths_mean": 0.0,
                "time_to_goal_mean": float(env.max_steps),
                "risk_integral_all": 0.0,
                "risk_integral_alive": 0.0,
                "collision_like": 0.0,
                "episode_len_mean": float(env.max_steps),
                "path_ratio": 1.0,
                "action_smoothness": 0.1,
                "energy_efficiency": 1.0,
                "safety_score": 1.0,
                "cost_progress": 0.0,
                "cost_risk": 0.0,
                "cost_wall": 0.0,
                "cost_collision": 0.0,
                "cost_energy": 0.0,
                "cost_jerk": 0.0,
                "cost_time": 0.0,
            }
            for scene in scenes
        }

    monkeypatch.setattr(tb, "_make_env", fake_make_env)
    monkeypatch.setattr(tb, "create_policy", fake_create_policy)
    monkeypatch.setattr(tb, "evaluate_policy", fake_evaluate_policy)
    tb._PERSISTENT_ENV_CACHE.clear()
    tb._PERSISTENT_POLICY_CACHE.clear()

    stages = [
        {
            "label": "warmup",
            "scenes": [{"id": "S0"}],
            "scene_ids": ["S0"],
            "scene_families": {"S0": "family_a"},
            "seeds": [1],
            "max_steps": 20,
        },
        {
            "label": "full",
            "scenes": [{"id": "S1"}],
            "scene_ids": ["S1"],
            "scene_families": {"S1": "family_b"},
            "seeds": [1, 2],
            "max_steps": 80,
        },
    ]

    for _ in range(2):
        tb._evaluate_candidate_progressive(
            policy_name="baseline:astar_grid",
            params={"alpha": 1.0},
            stages=stages,
            success_threshold=0.5,
            finish_min=0.7,
            alive_min=0.7,
            save_scenes=False,
            lite_metrics=True,
            env_payload={},
            goal_radius=3.0,
            aggregate_by_family=True,
            base_seed=0,
            persistent_runtime=True,
            policy_cache_size=4,
        )

    assert env_calls == [20, 80]
    assert policy_calls == [("baseline:astar_grid", 20, 0), ("baseline:astar_grid", 80, 0)]


def test_runtime_claim_detects_cross_thread_reuse(tmp_path):
    tb._RUNTIME_OWNERS.clear()
    debug_cfg = tb._parallel_debug_config(
        enabled=True,
        trace_enabled=True,
        trace_path=str(tmp_path / "parallel_trace.jsonl"),
        assert_ownership=True,
        allow_unsafe_parallel_optuna=True,
        allow_unsafe_persistent_runtime=True,
    )
    shared = object()
    start_second = threading.Event()
    finish_first = threading.Event()
    errors: list[Exception] = []

    def owner():
        with tb._runtime_claim(shared, kind="env", label={"trial": 1}, debug_cfg=debug_cfg):
            start_second.set()
            finish_first.wait(timeout=5)

    def intruder():
        start_second.wait(timeout=5)
        try:
            with tb._runtime_claim(shared, kind="env", label={"trial": 2}, debug_cfg=debug_cfg):
                pass
        except Exception as exc:
            errors.append(exc)
        finally:
            finish_first.set()

    t1 = threading.Thread(target=owner)
    t2 = threading.Thread(target=intruder)
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)

    assert errors
    assert "Нарушение владения runtime-объектом" in str(errors[0])
    trace_lines = (tmp_path / "parallel_trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert any("runtime_ownership_violation" in line for line in trace_lines)


def test_evaluate_policy_writes_parallel_trace(monkeypatch, tmp_path):
    def fake_run_episode(env, policy, scene, seed, success_threshold):
        return _stub_metrics()

    monkeypatch.setattr(tb, "run_episode", fake_run_episode)

    env = SwarmPZEnv(EnvConfig(), max_steps=10, goal_radius=3.0)
    policy = AStarGridPolicy()
    trace_path = tmp_path / "parallel_trace.jsonl"
    debug_cfg = tb._parallel_debug_config(
        enabled=True,
        trace_enabled=True,
        trace_path=str(trace_path),
        assert_ownership=True,
        allow_unsafe_parallel_optuna=False,
        allow_unsafe_persistent_runtime=False,
    )
    tb.evaluate_policy(
        env,
        policy,
        [{"id": "trace_scene"}],
        [1],
        success_threshold=0.5,
        finish_min=0.7,
        alive_min=0.7,
        cache_policy_name="baseline:astar_grid",
        cache_params={"alpha": 1.0},
        parallel_debug=debug_cfg,
        trial_number=7,
        stage_label="warmup",
    )
    lines = trace_path.read_text(encoding="utf-8").strip().splitlines()
    assert any("run_episode_start" in line for line in lines)
    assert any("policy_reset" in line for line in lines)
