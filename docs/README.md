# Индекс документации

Эта папка хранит документацию проекта, разделённую по назначению: текущее состояние системы, benchmark policy, runtime/infra, справочные материалы, журналы и perf-заметки.

## Рекомендуемый порядок чтения

### 1. Состояние проекта
- `README.md` — обзор проекта и быстрый старт
- `docs/showcase.md` — короткая витринная страница для внешнего читателя
- `docs/ROADMAP.md` — текущее состояние, приоритеты и завершённые workstream'ы
- `AGENTS.md` — краткая runtime-память для разработчика и coding agent'а

### 2. Бенчмарки и оценка
- `docs/benchmark_policy.md` — правила разделения на fair / privileged / oracle-only
- `docs/benchmarks/leaderboards.md` — канонические форматы результатных таблиц
- `docs/benchmarks/metric_semantics.md` — определения метрик и правила интерпретации

### 3. Runtime и инфраструктура
- `docs/infra/docker.md` — Docker/Compose runtime
- `docs/infra/hydra.md` — конфиги и модель запуска
- `docs/infra/mlflow.md` — tracking и политика хранения моделей
- `docs/infra/ui.md` — UI runtime и telemetry contracts

### 4. Справочные документы
- `docs/reference/env_schema.md` — контракт наблюдений
- `docs/reference/agent_playbook.md` — операционные правила для PPO / baseline / UI-изменений
- `docs/reference/diagnostics.md` — команды отладки и диагностические артефакты
- `docs/reference/model_storage.md` — правила хранения моделей
- `docs/reference/tech_debt.md` — только актуальный инженерный долг
- `common/README.md` — таксономия общих контрактов и compatibility layer для `common/`

### 5. Журналы
- `docs/logs/training_log.md` — проблемы и фиксы по PPO/curriculum
- `docs/logs/research_log.md` — заметки по архитектуре, tuning, UI и benchmark policy

### 6. Производительность
- `docs/perf/README.md` — эталонные fixed-step perf-снимки

## Правило обновления

Если изменение кода влияет на runtime-поведение, benchmark semantics или workflow, обновляй соответствующий документ в том же PR.
