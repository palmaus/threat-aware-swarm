# MLflow Tracking & Registry

MLflow — основной источник артефактов и метаданных запусков. Локальная папка `runs/` рассматривается как рабочий буфер, а не как окончательное хранилище.

## Быстрый старт

### Поднять сервис
```bash
docker compose up mlflow
```

### Запустить обучение
```bash
docker compose run --rm trainer python -m scripts.train.trained_ppo logging.tracking.mlflow.enabled=true experiment=ppo_waypoint
```

По умолчанию MLflow уже включён в `configs/hydra/logging/default.yaml`.

## Что логируется

| Категория | Содержимое |
| --- | --- |
| Конфиги | PPO/env/curriculum/resume/logging overrides |
| Метрики | train/eval метрики, `episode/*`, агрегаты протоколов |
| Системные метрики | CPU/RAM/GPU при `logging.tracking.system_metrics.enabled=true` |
| Артефакты | `meta/`, `models/`, `tb/`, отчёты и benchmark outputs |
| Теги | `best_fair`, `best_privileged`, `hero_demo` и прочие run tags |

## Source of truth policy
- если MLflow включён, локальный `runs/` можно очищать после успешной отправки артефактов;
- `logging.tracking.mlflow.cleanup_run_dir=true` уже включён по умолчанию;
- ключевые метаданные запуска должны жить в MLflow, а не только в локальной папке.

## Работа с моделями

### Оценка по `mlflow_run_id`
```bash
docker compose run --rm trainer python -m scripts.eval.eval_scenarios policy=ppo mlflow_run_id=<RUN_ID>
```

### Registry / aliases
В рантайме используй единый resolver `resolve_model_path()`. Он поддерживает:
- `mlflow:<run_or_model>`;
- `clearml:<task_id>`;
- локальные пути;
- алиасы `final|best|interrupt`.

## Когда нужен ClearML
ClearML остаётся опциональным. Его имеет смысл включать только если нужен внешний orchestration/UI поверх MLflow. Для обычного рабочего контура MLflow достаточно.

## Практические замечания
- `mlruns/` смонтирован в compose и хранит backend-state локального MLflow сервера.
- Если меняется структура входа модели, следи за `model_signature.json`.
- Для release/best-model workflows ориентируйся на MLflow tags и benchmark reports, а не на “ручной выбор zip-файла”.

## Связанные документы
- Docker runtime: `docs/infra/docker.md`
- Model storage: `docs/reference/model_storage.md`
- Benchmark policy: `docs/benchmark_policy.md`
