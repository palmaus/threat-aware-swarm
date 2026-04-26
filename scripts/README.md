# scripts/

CLI‑инструменты для обучения, оценки, анализа и воспроизводимых экспериментов.

## Основные команды

### Обучение
```bash
docker compose run --rm trainer python -m scripts.train.trained_ppo \
  experiment=ppo_waypoint
```

### Оценка
```bash
# оценка моделей
docker compose run --rm trainer python -m scripts.eval.eval_models \
  run_dir=runs/<run_id>

# оценка сценариев
docker compose run --rm trainer python -m scripts.eval.eval_scenarios \
  policy=baseline:astar_grid episodes=5

# сводка по run

docker compose run --rm trainer python -m scripts.eval.summarize_run \
  run_dir=runs/<run_id>
```

### Бенчмарки
```bash
docker compose run --rm trainer python -m scripts.bench.benchmark_baselines \
  policy=all n_episodes=10 max_steps=600
```

### Аналитика
```bash
# абляции
python3 -m scripts.analysis.ablation_runner \
  config=configs/experiments/ablation/lstm_vs_mlp.yaml

# eval protocol (seeds + CI)
python3 -m scripts.analysis.eval_protocol \
  policy=baseline:potential_fields seeds="[0,1,2,3,4]"

# экспорт ONNX
python3 -m scripts.analysis.export_onnx \
  model_path=runs/<run_id>/models/final.zip out_dir=runs/exports

# benchmarking suite
python3 -m scripts.analysis.benchmark_suite

# robustness suite
python3 -m scripts.analysis.robustness_suite

# demo report (обновление README-таблицы)
python3 -m scripts.analysis.demo_pack_report --update-readme
# опционально: PPO demo (требует DEMO_PPO_MODEL=/path/to/final.zip)
DEMO_PPO_MODEL=/path/to/final.zip make demo-pack
# график сравнения baseline vs PPO появится в docs/images/demo_metrics_compare.{png,svg}

# compare report (Markdown сравнение baseline vs PPO)
python3 -m scripts.analysis.compare_report

# demo story pack (копирование GIF в docs/images/demo_story/)
python3 -m scripts.analysis.story_pack --copy

# benchmark fair (static/dynamic)
python3 -m scripts.experiments --spec configs/experiments/benchmark_fair_static.yaml
python3 -m scripts.experiments --spec configs/experiments/benchmark_fair_dynamic.yaml

# benchmark privileged (static/dynamic)
python3 -m scripts.experiments --spec configs/experiments/benchmark_privileged_static.yaml
python3 -m scripts.experiments --spec configs/experiments/benchmark_privileged_dynamic.yaml
```

### Debug
```bash
python3 -m scripts.debug.debug_env_metrics steps=2000
python3 -m scripts.debug.finish_debug policy=baseline:potential_fields episodes=10
python3 -m scripts.debug.headless_rollout policy=baseline:flow_field steps=400 +oracle=privileged
python3 -m scripts.debug.calibrate_controller
python3 -m scripts.debug.health_report steps=500 agents=4
```

### Эксперименты по спеке
```bash
python3 -m scripts.experiments --spec configs/experiments/smoke.yaml --dry-run
make experiment EXPERIMENT_SPEC=configs/experiments/smoke.yaml
```

### Производительность
```bash
python3 -m scripts.perf.profile_env steps=2000
```

---

Все артефакты пишутся в `runs/` (или в MLflow при включённом трекинге).
Если меняется интерфейс или формат выходных данных — обновляйте этот README.
