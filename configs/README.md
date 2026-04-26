# configs/

Все конфиги проекта управляются через **Hydra**. Источник истины — `configs/hydra/`.

## Быстрый старт

```bash
docker compose run --rm trainer python -m scripts.train.trained_ppo \
  experiment=ppo_waypoint run.total_timesteps=20000 run.run_name=smoke
```

## Структура

```
configs/hydra/
  train.yaml
  env/default.yaml
  env_profile/{lite,full}.yaml
  ppo/default.yaml
  vec/default.yaml
  curriculum/{default,phase01,phase23,phase45,adr_alp}.yaml
  logging/default.yaml
  experiment/*.yaml
  eval/*.yaml
  bench/*.yaml
```

## Эксперименты

Оркестратор читает спеки из `configs/experiments/`:
```bash
python3 -m scripts.experiments --spec configs/experiments/smoke.yaml --dry-run
```

## Curriculum

- Основной файл: `configs/curriculum/base.yaml`
- Фазовые срезы: `configs/curriculum/phase01.yaml`, `phase23.yaml`, `phase45.yaml`
- Профили: `configs/hydra/curriculum/*`
- ADR/ALP профиль: `curriculum=adr_alp`

## Best‑params для бейзлайнов

Файл по умолчанию: `configs/best_policy_params.json`.
Переопределение:
```bash
POLICY_BEST_PARAMS=/path/to/best_policy_params.json
```

## Схема метрик

`configs/metrics_schema.yaml` — контракт метрик для eval/bench и golden‑сцен.

## Oracle visibility

Профили oracle (SSoT):
- `configs/hydra/oracle/fair.yaml` — no‑oracle input
- `configs/hydra/oracle/privileged.yaml` — oracle для baseline (map‑aware)
