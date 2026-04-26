# Tuning Presets

## Search profiles

Использование:
```bash
python -m scripts.tuning.tune_baselines tuning/profile=fast
python -m scripts.tuning.tune_baselines tuning/profile=balanced
python -m scripts.tuning.tune_baselines tuning/profile=deep
```

Профили:
- `tuning/profile=fast` — агрессивно дешёвый multi-fidelity search для ноутбука и smoke.
- `tuning/profile=balanced` — дефолтный режим, компромисс между скоростью и качеством отбора.
- `tuning/profile=deep` — более дорогой поиск перед финальным benchmark/update best params.

## Presets по длине эпизода

По умолчанию тюнер наследует `env.max_steps=600`, но это слишком мало для части
длинных сцен (`S8`, `S9`, `S10`, `S11`, `S14`, `S17`, `S18`, OOD-maze).
Чтобы не обрезать такие эпизоды на search/holdout этапе, добавлен отдельный
конфиг-групповой переключатель:

```bash
python -m scripts.tuning.tune_baselines tuning/step_budget=default
python -m scripts.tuning.tune_baselines tuning/step_budget=static
python -m scripts.tuning.tune_baselines tuning/step_budget=dynamic
python -m scripts.tuning.tune_baselines tuning/step_budget=long
```

Значения:
- `tuning/step_budget=default` → `env.max_steps=600`
- `tuning/step_budget=static` → `env.max_steps=900`
- `tuning/step_budget=dynamic` → `env.max_steps=1000`
- `tuning/step_budget=long` → `env.max_steps=1600`

Практически:
- для `scenes=[preset:static]` используй `tuning/step_budget=static`;
- для `scenes=[preset:dynamic]` используй `tuning/step_budget=dynamic`;
- для `S17/S18` и длинных OOD-сцен — `tuning/step_budget=long`.
- для ночного fair-тюнинга уже есть готовые root-level helper-скрипты:
  - `bash tune_fair_static.sh`
  - `bash tune_fair_dynamic.sh`
  - `bash night_fair_tune.sh`

Policy-aware нюанс:
- для тяжёлых global planners (`baseline:astar_grid`, `baseline:mpc_lite`) профиль `fast`
  автоматически использует более короткую двухстадийную схему;
- для остальных политик остаётся общий multi-stage preset.

Дополнительно:
- `eval_cache=true` — файловый кэш эпизодов (`policy+params+scene+seed+regime+env_hash`).
- `cache_version=v1` — ручной namespace для принудительной инвалидиации старого кэша.
- `cache_cleanup=true cache_ttl_days=14` — удаление старых cache-namespace по TTL.
- `persistent_workers=true` — reuse env/policy внутри long-lived workers и single-process search loop.

Важно:
- для `method=optuna n_jobs>1` тюнер теперь использует безопасный process-backend:
  главный процесс держит study/storage, а trial evaluation идёт в `spawn` worker'ах;
- внутри каждого worker'а `study.optimize(..., n_jobs=1)`, BLAS-потоки зажаты в `1`,
  а persistent runtime cache остаётся process-local;
- process-backend теперь автоматически режет число worker'ов по ресурсам через
  `parallel_max_workers`, `parallel_reserve_cpus`, `parallel_memory_fraction`;
  для тяжёлых planner'ов это защищает машину от перегруза по памяти;
- estimator памяти на worker теперь откалиброван по коротким probe-run'ам; старые
  значения были слишком консервативны и душили параллелизм, особенно у `astar_grid`;
- чтобы RSS не копился в долгоживущих `spawn`-process worker'ах, process-backend
  теперь recycle'ит процессы batch'ами и ограничивает число trial'ов на один
  worker через `parallel_max_trials_per_worker` (по умолчанию cap вычисляется
  автоматически от policy/profile/step budget);
- worker'ы стартуют через initializer с `PDEATHSIG=SIGTERM`; если родительский
  `tune_baselines` умирает, `spawn_main` не должен оставаться сиротой в системе;
  при exception/abort тюнер дополнительно добивает оставшихся `spawn_main`
  детей через `psutil`;
- дополнительно действует консервативный hard-cap по политикам:
  `astar_grid/mpc_lite/flow_field/potential_fields -> 8`;
- и отдельный cap на process-local `policy_cache_size`, чтобы persistent runtime
  не накапливал тяжёлые planner-объекты между trial'ами:
  `astar_grid -> 1`, `mpc_lite -> 2`, `flow_field/potential_fields -> 4`;
- debug-only thread path оставлен только через `allow_unsafe_parallel_optuna=true`
  и нужен лишь для controlled repro.

## Отладка внутренней параллельности

Для controlled repro добавлены флаги:
- `parallel_debug=true`
- `parallel_trace=true`
- `parallel_trace_path=...`
- `parallel_assert_ownership=true`

Для долгих прогонов, где нужно понять, что именно растёт по памяти, используй:

```bash
python -m scripts.debug.monitor_tuning_memory \
  --interval 1.0 \
  --csv runs/debug/tuning_mem.csv \
  --summary-json runs/debug/tuning_mem_summary.json \
  -- \
  python -m scripts.tuning.tune_baselines ...
```

Скрипт пишет:
- RSS главного процесса;
- суммарный RSS дерева процессов;
- число `spawn_main` worker'ов;
- системную used/available/cache memory и swap.
- `allow_unsafe_parallel_optuna=true`
- `allow_unsafe_persistent_runtime=true`

Что они делают:
- пишут structured trace в `parallel_trace.jsonl`;
- ставят ownership-asserts на runtime-объекты `env/policy`;
- позволяют временно вернуть unsafe thread-based path только для диагностики.

Пример controlled repro:
```bash
NUMBA_DISABLE_JIT=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python -m scripts.tuning.tune_baselines \
  policy=['baseline:astar_grid'] \
  tuning/profile=fast \
  method=optuna trials=2 n_jobs=2 pruner=none episodes=1 \
  parallel_debug=true parallel_trace=true \
  allow_unsafe_parallel_optuna=true \
  allow_unsafe_persistent_runtime=true \
  scenes=[scenarios/S0_sanity_no_threats.yaml]
```

Диагностический вывод:
- если trace показывает `runtime_ownership_violation`, значит два thread'а вошли
  в один shared `SwarmPZEnv`/policy;
- это и есть подтверждённый Python-level race, из-за которого shared runtime cache
  запрещён для threaded Optuna.

Рабочий режим:
- для обычного тюнинга эти unsafe-флаги не нужны;
- достаточно указать `method=optuna n_jobs=<число воркеров>`, и тюнер сам выберет
  process-backend со `spawn`.

## Selection semantics

Тюнер теперь сохраняет не только `balanced/safe/fast`, но и `stable` champion:
- `balanced` — лучший по общей objective с учётом uncertainty tie-break;
- `safe` — минимум риска при соблюдении gates;
- `fast` — минимум времени;
- `stable` — минимум variability (`*_std`) между эпизодами/сценами.

В fair-режиме тюнер дополнительно режет policy-level privileged params
(`oracle`, `full_map`, `privileged`, `map_aware` и т.п.), даже если они были
подсунуты через search space вручную.

## Scoring presets

Использование:
```bash
python -m scripts.tuning.tune_baselines scoring=fast
python -m scripts.tuning.update_best_params scoring=aggressive
```

Пресеты:
- `scoring=default` — сбалансированно: finish/alive как основа.
- `scoring=fast` — скорость важнее риска.
- `scoring=aggressive` — безопасность важнее скорости.

## Tuning spaces

Шаблоны параметров находятся в `configs/hydra/tuning/spaces/`.
Пример:
```bash
python -m scripts.tuning.tune_baselines \
  tuning/profile=balanced \
  tuning/step_budget=dynamic \
  policy=['baseline:mpc_lite'] \
  space=configs/hydra/tuning/spaces/mpc_lite.yaml \
  method=optuna trials=50 stage_b=true topk=5
```

Make aliases:
```bash
make tune-fast TUNE_OVERRIDES="policy=['baseline:mpc_lite'] scenes=[preset:static] tuning/step_budget=static"
make tune-balanced TUNE_OVERRIDES="policy=[all] scenes=[preset:static] tuning/step_budget=static method=optuna trials=80 n_jobs=8 pruner=asha stage_b=true"
make tune-deep TUNE_OVERRIDES="policy=[all] scenes=[preset:dynamic] tuning/step_budget=dynamic episodes=4 episodes_eval=12"
```

## Smoke tests

- Обычный `pytest -q` и `pytest -q -m integration` теперь пропускают тяжёлый
  holdout-validating smoke.
- Чтобы прогнать его локально, используй:

```bash
pytest -q -m tuning_stageb --run-tuning-stageb
# или
make test-tuning-stageb
```

- В CI этот smoke вынесен в отдельный job `tuning-stageb`, чтобы не тормозить
  основной unit/integration suite.

## Observed wall-clock (local smoke)

Снято локально на `AMD Ryzen 7 4800HS`, артефакт: `runs/perf_tuning_profiles_20260306/timings.json`.

Протокол замера:
- `method=random`
- `samples=1`
- `n_jobs=1`
- `episodes=1`
- `stage_b=false`
- `scenes=[scenarios/S0_sanity_no_threats.yaml, scenarios/S18_hard_maze.yaml]`
- `information_regime=fair`

Результаты:

| policy | profile | cache | wall-clock, s |
|---|---|---|---:|
| `baseline:astar_grid` | `fast` | cold | 37.4 |
| `baseline:astar_grid` | `balanced` | cold | 36.6 |
| `baseline:astar_grid` | `balanced` | warm | 19.4 |
| `baseline:potential_fields` | `fast` | cold | 17.6 |
| `baseline:potential_fields` | `balanced` | cold | 17.8 |
| `baseline:potential_fields` | `balanced` | warm | 10.5 |

Выводы:
- `eval_cache` даёт заметный выигрыш на повторном прогоне: примерно `-47%` для `astar_grid` и `-41%` для `potential_fields`.
- В search-only smoke `tuning/profile=balanced` оказался не медленнее `fast`; причина в том, что `fast` добавляет промежуточный `focus` stage. Поэтому `fast` стоит рассматривать как профиль для более агрессивного multi-stage search, а не как гарантированно более быстрый режим на каждом сценарии.
- Для повседневной работы оставляем `balanced` как дефолтный профиль; `fast` имеет смысл прежде всего для Optuna/TPE + pruning, а `deep` — перед финальным обновлением best params.

## Observed wall-clock (local holdout-validating smoke)

Снято локально на `AMD Ryzen 7 4800HS`, артефакт: `runs/perf_tuning_stageb_20260306/timings.json`.

Протокол замера:
- `tuning/profile=balanced`
- `method=optuna`
- `trials=1`
- `n_jobs=1`
- `pruner=none`
- `episodes=1`
- `episodes_eval=1`
- `stage_b=true`
- `topk=1`
- `search=S0`, `holdout=S18`, `benchmark=S1`, `ood=S11`

Результаты:

| policy | cache | wall-clock, s |
|---|---|---:|
| `baseline:astar_grid` | cold | 48.7 |
| `baseline:astar_grid` | warm | 40.4 |
| `baseline:potential_fields` | cold | 23.6 |
| `baseline:potential_fields` | warm | 17.6 |

Выводы:
- Stage-B/benchmark/OOD validation добавляет заметную стоимость по сравнению с search-only smoke.
- `eval_cache` всё ещё окупается, но слабее, чем в search-only режиме: примерно `-17%` для `astar_grid` и `-26%` для `potential_fields`.
- Для рабочего цикла это подтверждает разделение: aggressive search делать на search-only бюджете, а holdout-validating запускать реже — на shortlist кандидатов.

## Guarded helper scripts

Локальные helper-скрипты:
- `bash tune_fair_static.sh`
- `bash tune_fair_dynamic.sh`
- `bash night_fair_tune.sh`

Для `baseline:astar_grid` они теперь по умолчанию включают memory guard:
- запуск идёт через `debug_tuning_memory_scope.sh`;
- внутри используется `systemd-run --user --scope` с лимитом памяти;
- мониторинг пишет `*_mem.csv` и `*_mem_summary.json` в `runs/night_tune_logs/`.

Полезные env-переменные:
- `ASTAR_MEMORY_GUARD=0` — отключить guard полностью;
- `ASTAR_MEMORY_MAX=8G` — лимит памяти для `astar_grid`;
- `ASTAR_TIMEOUT=28800` — timeout guarded-run;
- `ASTAR_MONITOR_INTERVAL=1.0` — интервал съёма RSS.
