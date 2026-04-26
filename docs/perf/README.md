# Эталонные perf-снимки

Эта папка хранит эталонные perf-снимки, которые используются для ловли регрессий после изменений в среде или baseline-слое.

## Назначение
- держать хотя бы один стабильный fixed-step perf-бейзлайн;
- сравнивать будущие оптимизации с фиксированным сценарием;
- отделять разговор о производительности от разговора о качестве алгоритма.

## Текущий эталонный снимок

| Поле | Значение |
| --- | --- |
| Scenario | `scenarios/S7_dynamic_chaser.yaml` |
| Steps | `600` fixed steps |
| Early stop | disabled |
| Артефакт | `docs/perf/baseline_profile_dynamic_600_fixed.json` |
| Сгенерирован | `2026-03-05` |

## Время на шаг

| Baseline | s/step |
| --- | ---: |
| `baseline:astar_grid` | 0.01191 |
| `baseline:brake` | 0.00419 |
| `baseline:flow_field` | 0.00432 |
| `baseline:greedy` | 0.00394 |
| `baseline:greedy_safe` | 0.00464 |
| `baseline:mpc_lite` | 0.01129 |
| `baseline:potential_fields` | 0.00436 |
| `baseline:random` | 0.00401 |
| `baseline:separation_steering` | 0.00606 |
| `baseline:wall` | 0.00395 |
| `baseline:zero` | 0.00329 |

## 2026-03-06 — повторная проверка после env runtime pass

Это сравнение использует тот же сценарий и те же `600` fixed steps, но
официальным reference остаётся файл от `2026-03-05`. Здесь важно различать две
точки зрения:

- `end-to-end` — включает setup policy/env и one-time warmup;
- `step-only` — сравнивает steady-state runtime для фиксированного 600-step rollout.

Для planner-heavy baseline'ов корректным сигналом регрессии является именно
`step-only`. В частности, `baseline:mpc_lite` сейчас платит заметный one-time
setup cost, поэтому end-to-end число выглядит хуже, хотя steady-state runtime
ускорился.

Артефакты:
- `runs/perf_compare_dynamic_600_20260306/current_profile_dynamic_600.json`
- `runs/perf_compare_dynamic_600_20260306/step_only_profile_dynamic_600.json`
- `runs/perf_compare_dynamic_600_20260306/summary.json`
- `runs/perf_compare_dynamic_600_20260306/summary.md`

### Delta `step-only` относительно reference

| Baseline | Old s/step | New s/step | Delta % | Speedup x |
| --- | ---: | ---: | ---: | ---: |
| `baseline:random` | 0.00401 | 0.00372 | -7.2% | 1.08 |
| `baseline:zero` | 0.00329 | 0.00308 | -6.4% | 1.07 |
| `baseline:greedy` | 0.00394 | 0.00297 | -24.7% | 1.33 |
| `baseline:greedy_safe` | 0.00464 | 0.00420 | -9.5% | 1.11 |
| `baseline:wall` | 0.00395 | 0.00278 | -29.7% | 1.42 |
| `baseline:brake` | 0.00419 | 0.00278 | -33.7% | 1.51 |
| `baseline:separation_steering` | 0.00606 | 0.00407 | -32.8% | 1.49 |
| `baseline:potential_fields` | 0.00436 | 0.00404 | -7.4% | 1.08 |
| `baseline:flow_field` | 0.00432 | 0.00409 | -5.3% | 1.06 |
| `baseline:astar_grid` | 0.01191 | 0.01147 | -3.7% | 1.04 |
| `baseline:mpc_lite` | 0.01129 | 0.01030 | -8.8% | 1.10 |

### Комментарии

- `env_zero_direct` probe: `1.388s total`, `0.00231 s/step`.
- у `baseline:mpc_lite` примерно `2.10s` one-time setup/warmup overhead
  в end-to-end прогоне, поэтому регрессии нужно оценивать по `step-only`.
- steady-state у `baseline:astar_grid` тоже улучшился; оставшаяся цена сидит
  в локальной planner-логике, а не в hot path среды.

## 2026-03-06 — финальная проверка recovery для planner-layer

После follow-up recovery pass для `A*` и `MPC-lite` fixed-step проверка на
`scenarios/S7_dynamic_chaser.yaml` была повторена в двух представлениях:

- `wall_time_s_per_step` — реальный fixed-step runtime без profiler overhead;
- `step_only_s_per_step` — `cProfile` runtime для hotspot-анализа.

Артефакты:
- `runs/perf_astar_mpc_final_20260306/profile.json`
- `runs/perf_astar_mpc_final_20260306/summary.md`

### Финальные числа

| Policy | cProfile s/step | wall s/step |
| --- | ---: | ---: |
| `baseline:astar_grid` | 0.02696 | 0.00635 |
| `baseline:mpc_lite` | 0.02046 | 0.00473 |

### Интерпретация

- recovery-фиксы ещё раз улучшили planner hot paths, особенно reuse path в A*
  и batch-controller path в `mpc_lite`;
- `cProfile` теперь заметно завышает реальную steady-state цену для
  planner-heavy policy, потому что profiler платит за каждый Python helper call;
- для regression-решений нужно смотреть на wall-clock, а `cProfile` оставлять
  только для ranking hot spots.

## 2026-03-06 — контрольный прогон после стабилизации документации и `common/`

После закрытия `PR-STAB-4` и `PR-STAB-5` fixed-step проверка была повторена тем
же официальным profiler'ом уже через regression-gate against
`runs/perf_astar_mpc_final_20260306/profile.json`.

Артефакты:
- `runs/perf_regression_recheck_20260306/profile.json`
- `runs/perf_regression_recheck_20260306/comparison.json`
- `runs/perf_regression_recheck_20260306/summary.md`

Итог по wall-clock:

| Policy | Эталон, s/step | Новый прогон, s/step | Delta % |
| --- | ---: | ---: | ---: |
| `baseline:astar_grid` | 0.00635 | 0.00605 | -4.7% |
| `baseline:mpc_lite` | 0.00473 | 0.00467 | -1.4% |

Вывод:
- perf-regression после закрытия старых PR нет;
- официальный `scripts.perf.profile_baselines` снова рабочий после фикса
  `ensure_run_dir(..., run_id=None)`;
- текущая эталонная точка для planner-layer остаётся валидной.

## Как использовать эту папку
- воспринимай эти числа как эталон регрессии, а не как метрику качества задачи;
- после perf-sensitive изменений переснимай профиль и сравнивай с эталоном;
- если новый snapshot становится основным, замени reference-файл и обнови эту страницу.

## Официальный profiler / gate

Используй fixed-step profiler вместо ad-hoc сниппетов:

```bash
python3 -m scripts.perf.profile_baselines \
  --scenario scenarios/S7_dynamic_chaser.yaml \
  --steps 600 \
  --policies baseline:astar_grid baseline:mpc_lite
```

Проверяй регрессии относительно reference snapshot по wall-clock:

```bash
python3 -m scripts.perf.profile_baselines \
  --scenario scenarios/S7_dynamic_chaser.yaml \
  --steps 600 \
  --policies baseline:astar_grid baseline:mpc_lite \
  --compare docs/perf/baseline_profile_dynamic_600_fixed.json \
  --metric wall \
  --max-regression-pct 10
```

Make-таргеты:
- `make profile-baselines`
- `make perf-gate`

## Related docs
- `docs/reference/tech_debt.md`
- `docs/logs/research_log.md`
