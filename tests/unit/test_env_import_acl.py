"""Проверка, что ядро env не зависит от UI/скриптов."""

from __future__ import annotations

import ast
from pathlib import Path


def test_env_has_no_ui_or_scripts_imports() -> None:
    env_dir = Path(__file__).resolve().parents[2] / "env"
    offenders: list[Path] = []
    for path in env_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and (node.module.startswith("ui") or node.module.startswith("scripts")):
                    offenders.append(path)
                    break
            if isinstance(node, ast.Import):
                for name in node.names:
                    if name.name.startswith("ui") or name.name.startswith("scripts"):
                        offenders.append(path)
                        break
    assert not offenders, f"Запрещён импорт ui/scripts в env: {offenders}"


def test_runtime_uses_nested_common_imports() -> None:
    root = Path(__file__).resolve().parents[2]
    legacy_modules = {
        "common.context",
        "common.contracts",
        "common.oracle_visibility",
        "common.physics_model",
    }
    scan_roots = [root / "env", root / "baselines", root / "scripts", root / "ui"]
    offenders: list[str] = []
    for scan_root in scan_roots:
        for path in scan_root.rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            if rel in legacy_modules:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module in legacy_modules:
                    offenders.append(rel)
                    break
                if isinstance(node, ast.Import):
                    if any(alias.name in legacy_modules for alias in node.names):
                        offenders.append(rel)
                        break
    assert not offenders, f"Runtime код должен импортировать nested common-пути: {offenders}"


def test_ui_controller_does_not_reach_into_env_sim() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "ui" / "controller.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr != "sim":
            continue
        parent = node.value
        if isinstance(parent, ast.Name) and parent.id == "env":
            offenders.append(node.lineno)
        if isinstance(parent, ast.Attribute) and parent.attr == "env":
            offenders.append(node.lineno)
    assert not offenders, f"ui/controller.py должен использовать public env API вместо env.sim: {offenders}"


def test_common_policy_does_not_import_env_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    common_policy = root / "common" / "policy"
    offenders: list[str] = []
    for path in common_policy.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("env"):
                offenders.append(path.relative_to(root).as_posix())
                break
            if isinstance(node, ast.Import):
                if any(alias.name.startswith("env") for alias in node.names):
                    offenders.append(path.relative_to(root).as_posix())
                    break
    assert not offenders, f"common.policy не должен зависеть от env runtime: {offenders}"


def test_runtime_policy_entrypoints_use_shared_baseline_factory() -> None:
    root = Path(__file__).resolve().parents[2]
    entrypoints = [
        root / "scripts" / "bench" / "benchmark_baselines.py",
        root / "scripts" / "debug" / "finish_debug.py",
        root / "scripts" / "eval" / "eval_scenarios.py",
        root / "scripts" / "perf" / "profile_baselines.py",
        root / "scripts" / "tuning" / "policy_factory.py",
        root / "ui" / "policies.py",
        root / "ui" / "policy_workers.py",
    ]
    offenders: list[str] = []
    missing_factory: list[str] = []
    for path in entrypoints:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        rel = path.relative_to(root).as_posix()
        imports_factory = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "baselines.factory":
                    imports_factory = True
                if node.module == "scripts.common.scenario_eval":
                    imports_factory = True
                if node.module == "baselines.policies" and any(alias.name == "default_registry" for alias in node.names):
                    offenders.append(rel)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "baselines.factory":
                        imports_factory = True
                    if alias.name == "scripts.common.scenario_eval":
                        imports_factory = True
                    if alias.name == "baselines.policies.default_registry":
                        offenders.append(rel)
        if not imports_factory:
            missing_factory.append(rel)
    assert not missing_factory, f"Entry-points должны использовать baselines.factory: {missing_factory}"
    assert not offenders, f"Entry-points не должны напрямую создавать registry: {offenders}"


def test_runtime_entrypoints_use_shared_env_factory() -> None:
    root = Path(__file__).resolve().parents[2]
    entrypoints = [
        root / "scripts" / "bench" / "bench_obs_perf.py",
        root / "scripts" / "bench" / "benchmark_baselines.py",
        root / "scripts" / "debug" / "debug_env_metrics.py",
        root / "scripts" / "debug" / "finish_debug.py",
        root / "scripts" / "debug" / "health_report.py",
        root / "scripts" / "eval" / "eval_scenarios.py",
        root / "scripts" / "eval" / "eval_models.py",
        root / "scripts" / "perf" / "profile_baselines.py",
        root / "scripts" / "perf" / "profile_env.py",
        root / "scripts" / "train" / "trained_ppo.py",
        root / "scripts" / "tuning" / "tune_baselines.py",
        root / "ui" / "controller.py",
    ]
    missing: list[str] = []
    forbidden_construction: list[str] = []
    for path in entrypoints:
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        if "common.runtime.env_factory" not in source:
            missing.append(rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "SwarmPZEnv":
                forbidden_construction.append(f"{rel}:{node.lineno}")
    assert not missing, f"Entry-points должны использовать common.runtime.env_factory: {missing}"
    assert not forbidden_construction, f"Entry-points не должны напрямую конструировать SwarmPZEnv: {forbidden_construction}"


def test_eval_debug_use_shared_rl_episode_runner() -> None:
    root = Path(__file__).resolve().parents[2]
    entrypoints = [
        root / "scripts" / "debug" / "headless_rollout.py",
        root / "scripts" / "eval" / "eval_models.py",
        root / "scripts" / "eval" / "eval_scenarios.py",
    ]
    missing: list[str] = []
    for path in entrypoints:
        source = path.read_text(encoding="utf-8")
        if "scripts.common.rl_episode_runner" not in source and "scripts.common.scenario_eval" not in source:
            missing.append(path.relative_to(root).as_posix())
    assert not missing, f"Eval/debug PPO paths должны переиспользовать rl_episode_runner: {missing}"


def test_analysis_debug_tuning_use_shared_scenario_eval_layer() -> None:
    root = Path(__file__).resolve().parents[2]
    entrypoints = [
        root / "scripts" / "analysis" / "benchmark_suite.py",
        root / "scripts" / "analysis" / "eval_protocol.py",
        root / "scripts" / "analysis" / "robustness_suite.py",
        root / "scripts" / "debug" / "headless_rollout.py",
        root / "scripts" / "tuning" / "tune_baselines.py",
    ]
    missing: list[str] = []
    legacy_imports: list[str] = []
    for path in entrypoints:
        rel = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        if "scripts.common.scenario_eval" not in source:
            missing.append(rel)
        if "scripts.eval.eval_scenarios" in source:
            legacy_imports.append(rel)
    assert not missing, f"Shared scenario helpers должны жить вне eval entrypoint: {missing}"
    assert not legacy_imports, f"Analysis/debug/tuning не должны импортировать eval_scenarios как service-layer: {legacy_imports}"


def test_ui_policy_workers_use_factory_not_concrete_planners() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "ui" / "policy_workers.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_imports: list[int] = []
    has_factory = "baselines.factory" in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"baselines.astar_grid", "baselines.mpc_lite"}:
            forbidden_imports.append(node.lineno)
    assert has_factory
    assert not forbidden_imports, f"UI workers не должны импортировать concrete planner classes: {forbidden_imports}"


def test_spawn_oracle_use_public_engine_views() -> None:
    root = Path(__file__).resolve().parents[2]
    forbidden = {
        root / "env" / "scenes" / "spawn_controller.py": (
            "._env.sim",
            "._env._walls",
            "._env._static_circles",
            "._env._threats",
        ),
        root / "env" / "oracles" / "manager.py": (
            "._env.sim",
            "._env._path_len",
            "._env._threats",
        ),
    }
    offenders: list[str] = []
    for path, patterns in forbidden.items():
        source = path.read_text(encoding="utf-8")
        rel = path.relative_to(root).as_posix()
        for pattern in patterns:
            if pattern in source:
                offenders.append(f"{rel}:{pattern}")
    assert not offenders, f"Spawn/oracle должны читать runtime через public engine views: {offenders}"


def test_tuning_report_and_eval_cache_are_split_from_orchestrator() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "tuning" / "tune_baselines.py"
    source = path.read_text(encoding="utf-8")
    assert "from scripts.tuning import eval_cache as eval_cache_helpers" in source
    assert "from scripts.tuning import reporting as tuning_reporting" in source
    assert "shutil.rmtree" not in source
    assert "json_path = out_dir / \"tuning_report.json\"" not in source


def test_mpc_lite_uses_shared_fallbacks_and_wall_kernels() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "baselines" / "mpc_lite.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden_defs = {"_circle_rect_normal_numba", "_circle_hits_any_numba", "_resolve_wall_slide_numba"}
    has_walls_numba_import = False
    direct_astar_imports: list[int] = []
    duplicate_wall_defs: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "common.physics.walls_numba":
                has_walls_numba_import = True
            if node.module == "baselines.astar_grid":
                direct_astar_imports.append(node.lineno)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in forbidden_defs:
            duplicate_wall_defs.append(node.lineno)
    assert has_walls_numba_import
    assert not direct_astar_imports, f"MPC не должен напрямую импортировать A*: {direct_astar_imports}"
    assert not duplicate_wall_defs, f"MPC не должен дублировать wall numba kernels: {duplicate_wall_defs}"


def test_scene_providers_use_engine_public_mutators() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "env" / "scenes" / "providers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    private_mutations: list[int] = []
    direct_sim_access: list[int] = []
    concrete_engine_imports: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "env.engine":
            concrete_engine_imports.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            if isinstance(node.value, ast.Name) and node.value.id == "env":
                private_mutations.append(node.lineno)
        if isinstance(node, ast.Attribute) and node.attr == "sim":
            if isinstance(node.value, ast.Name) and node.value.id == "env":
                direct_sim_access.append(node.lineno)
    assert not concrete_engine_imports, f"providers не должен импортировать concrete engine: {concrete_engine_imports}"
    assert not private_mutations, f"providers должен идти через public mutators: {private_mutations}"
    assert not direct_sim_access, f"providers не должен обращаться к env.sim напрямую: {direct_sim_access}"


def test_scene_manager_uses_engine_snapshot_api() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "env" / "scenes" / "scene_manager.py"
    source = path.read_text(encoding="utf-8")
    forbidden = ["env.sim", "._set_field_size", "._target_vel", "._target_motion", "._walls", "._static_circles"]
    offenders = [pattern for pattern in forbidden if pattern in source]
    assert "capture_runtime_snapshot" in source
    assert "restore_runtime_snapshot" in source
    assert not offenders, f"scene_manager должен идти через public snapshot/mutator API engine: {offenders}"


def test_debug_metrics_use_public_runtime_views_and_shared_vec_wrapper() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "debug" / "debug_env_metrics.py"
    source = path.read_text(encoding="utf-8")
    assert "pettingzoo_to_vec_env" in source
    assert "import supersuit as ss" not in source
    assert "env.sim" not in source


def test_env_factory_keeps_supersuit_lazy_for_ui_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "common" / "runtime" / "env_factory.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    eager_supersuit: list[int] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            if any(alias.name == "supersuit" for alias in node.names):
                eager_supersuit.append(node.lineno)
        if isinstance(node, ast.ImportFrom) and node.module == "supersuit":
            eager_supersuit.append(node.lineno)
    assert not eager_supersuit, f"env_factory не должен eagerly импортировать supersuit на module import: {eager_supersuit}"
