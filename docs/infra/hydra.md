# Hydra Runtime

Hydra — единственный поддерживаемый способ запуска train/eval/bench/debug/tuning сценариев.

## Source of truth
- все runtime-конфиги живут в `configs/hydra/`;
- overrides идут только через Hydra;
- старый argparse/legacy CLI в рабочем контуре не используется.

## Основные группы конфигов

| Группа | Что задаёт |
| --- | --- |
| `env` | физика, награды, шумы, oracle, сценарии |
| `env_profile` | лёгкий или полный профиль среды (`lite|full`) |
| `ppo` | архитектура и PPO hyperparams |
| `vec` | векторизованные env и worker layout |
| `curriculum` | staged curriculum, ADR/ALP |
| `resume` | resume-поведение и источники моделей |
| `logging` | MLflow/ClearML/system metrics |
| `experiment` | готовые пресеты запуска |
| `ui` | параметры web UI |
| `tuning` | протокол тюнинга и scoring |

## Типовые команды

### Train
```bash
docker compose run --rm trainer python -m scripts.train.trained_ppo experiment=ppo_waypoint run.total_timesteps=4000000
```

### Resume
```bash
docker compose run --rm trainer python -m scripts.train.trained_ppo experiment=resume resume.enabled=true resume.run_dir=latest:phase23
```

### Multirun
```bash
docker compose run --rm trainer python -m scripts.train.trained_ppo -m experiment=cnn_heavy run.seed=0,1,2 hydra/launcher=joblib
```

### Eval
```bash
docker compose run --rm trainer python -m scripts.eval.eval_scenarios policy=baseline:astar_grid scenes=[preset:sanity]
```

### Tuning
```bash
docker compose run --rm trainer python -m scripts.tuning.tune_baselines tuning/profile=balanced policy=[all]
```

## Что сохраняется автоматически
- `runs/<run_id>/.hydra/config.yaml` — сырой Hydra dump;
- `runs/<run_id>/meta/hydra_config.yaml` — копия для пайплайна и артефактов;
- `config_audit.json` — для eval/bench/tune/protocol сценариев;
- `manifest.json` / `metric_manifest.json` — для reproducibility и metric governance.

## Правила изменения конфигов
- новые runtime-параметры должны жить в dataclass/structured config, а не появляться “мимо” конфигов;
- если параметр влияет на env runtime, он должен быть учтён либо в `EnvConfig`, либо в application layer (`apply_curriculum`, UI dispatcher, tuning protocol);
- если меняется структура метрик, обновляй `configs/metrics_schema.yaml` и связанные документы.

## Связанные документы
- Контейнерный запуск: `docs/infra/docker.md`
- Tracking: `docs/infra/mlflow.md`
- Тюнинг: `configs/hydra/tuning/README.md`
