# Docker & Compose

Этот документ описывает рабочий контейнерный контур проекта. По умолчанию проект предполагает запуск через `docker compose`.

## Основные сервисы

| Сервис | Назначение | Команда |
| --- | --- | --- |
| `trainer` | обучение, eval, tuning, analysis | `docker compose run --rm trainer ...` |
| `web` | FastAPI + React UI | `docker compose up web` |
| `mlflow` | tracking UI и registry backend | `docker compose up mlflow` |
| `tensorboard` | просмотр TB-логов | `docker compose up tensorboard` |
| `jupyter` | ноутбуки и ad-hoc анализ | `docker compose up jupyter` |
| `cli` | интерактивная shell-сессия в контейнере | `docker compose run --rm cli` |

## Рекомендуемый путь запуска

### Сборка
```bash
docker compose build trainer
```

### Обучение
```bash
docker compose run --rm trainer python -m scripts.train.trained_ppo experiment=ppo_waypoint
```

### Оценка сценариев
```bash
docker compose run --rm trainer python -m scripts.eval.eval_scenarios policy=baseline:potential_fields episodes=5
```

### UI
```bash
docker compose up web
# http://127.0.0.1:8000
```

### MLflow
```bash
docker compose up mlflow
# http://127.0.0.1:5000
```

### TensorBoard
```bash
docker compose up tensorboard
# http://127.0.0.1:6006
```

## Raw Docker путь

Используй его только если нужно собирать отдельные CPU/GPU образы вручную.

### CPU
```bash
docker build -t threat-aware-swarm:cpu -f Dockerfile .
```

### GPU
```bash
docker build -t threat-aware-swarm:gpu -f Dockerfile.gpu .
```

Для GPU нужен NVIDIA Container Toolkit.

## Практические замечания
- Артефакты хранятся на хосте в `runs/`, `mlruns/` и связанных volume.
- Runtime cache монтируется в `/app/.cache`; это важно для Numba/Torch/MPL.
- Для корректных прав на файлы используй `.env` на основе `.env.example`.
- При изменении зависимостей пересобирай `trainer`.

## Частые проблемы

### Нет доступа к Docker socket
Проверь права на `docker.sock` или членство в группе `docker`.

### После изменения UI виден старый фронтенд
Пересобери фронтенд:
```bash
make ui-build
```

### Долгая сборка образа
Используй BuildKit:
```bash
export DOCKER_BUILDKIT=1
docker compose build trainer
```

## Связанные документы
- Конфиги и overrides: `docs/infra/hydra.md`
- Tracking и артефакты: `docs/infra/mlflow.md`
- Web UI: `docs/infra/ui.md`
