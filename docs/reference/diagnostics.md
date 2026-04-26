# Diagnostics Guide

Этот документ собирает быстрые команды и артефакты для разбора типовых проблем: пропавшие метрики, нулевой risk, отсутствие finish и расхождения между env и benchmark.

## Быстрые команды

### Диагностика benchmark metrics
```bash
docker compose run --rm trainer python -m scripts.bench.benchmark_baselines   policy=all n_episodes=2 max_steps=200 seed=0 debug_metrics=true
```

### Сравнение env vs wrappers
```bash
docker compose run --rm trainer python -m scripts.debug.debug_env_metrics   max_steps=200 seed=0
```

### Finish debug
```bash
docker compose run --rm trainer python -m scripts.debug.finish_debug   policy=baseline:potential_fields episodes=20 seed=0
```

## Основные артефакты

| Артефакт | Где появляется | Что смотреть |
| --- | --- | --- |
| `examples_infos.json` | `runs/diagnostics/` | первые infos по политикам |
| `counts.json` | `runs/diagnostics/` | сколько раз ключи пропали |
| `infos_schema.md` | `runs/diagnostics/` | частота ключей по политикам |
| `risk_timeline_*.jsonl` | `runs/diagnostics/` | risk до/после шага, смерти, дистанции |
| `env_metrics.json` | `runs/diagnostics/` | env vs wrappers breakdown |
| `geometry_stats_*` / `episode_geometry_*` | `runs/diagnostics/` | геометрия сцены и близость угроз |
| `finish_debug_*` / `finish_trace_*` | `runs/diagnostics/` | дистанции до цели и удержание в зоне |

## Как интерпретировать симптомы

### `risk_p` есть в env, но пропадает в отчёте
Ищи проблему в wrapper/aggregation контуре.

### `risk_p` уже нулевой внутри env
Проверяй генерацию угроз, интенсивности и sanity-сценарий `debug_env_metrics`.

### Агент доходит до цели, но нет finish
Проверяй `goal_hold_steps`, `max_in_goal_streak`, `min_dist_to_target`.

### В `death_events` есть смерти, а risk нулевой
Смотри `risk_timeline_*` и сравни `risk_p_before` vs `risk_p_after`.

## Полезные поля в диагностике
- `died_this_step`
- `dist_to_nearest_threat`
- `nearest_threat_margin`
- `nearest_threat_intensity`
- `inside_nearest` / `inside_any`
- `min_wall_dist`

Эти поля помогают понять, смерть связана с угрозой, стеной или общей деградацией policy.

## Sanity-check
Если `debug_env_metrics` ставит агента внутрь радиуса угрозы, а `risk_p` всё равно не становится положительным, это уже env-level regression.
