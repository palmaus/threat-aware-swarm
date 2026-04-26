# baselines/

Эта папка содержит эвристические политики и планировщики. Они не требуют обучения, запускаются быстро и дают стабильные ориентиры качества.

## Что внутри

- **A* Grid (Memory)** — локальный A* с памятью стен/угроз и эвристиками выхода из ловушек.
- **MPC‑Lite (CEM)** — короткий роллаут с оптимизацией управления и мягким риском.
- **Potential Fields** — силы притяжения/отталкивания, демпфирование и скольжение вдоль стен.
- **Flow Field (privileged)** — map‑aware reference: глобальное поле расстояний (Dijkstra), использует full static map.
- **Oracle (eval‑only)** — эталон пути по статике для метрики `path_ratio` (не baseline).

Все бейзлайны работают через единый конвейер `Perception → Planner → Controller` и используют `WaypointController`.

## Быстрый запуск

```bash
# Все бейзлайны
docker compose run --rm trainer python -m scripts.bench.benchmark_baselines \
  policy=all n_episodes=10 max_steps=600 seed=0

# Один бейзлайн (privileged reference)
docker compose run --rm trainer python -m scripts.eval.eval_scenarios \
  policy=baseline:flow_field episodes=20 +oracle=privileged

# Fair‑режим (no oracle input)
docker compose run --rm trainer python -m scripts.eval.eval_scenarios \
  policy=baseline:mpc_lite episodes=20 env.oracle_visibility=none
```

## Тюнинг

```bash
# Запуск Optuna
docker compose run --rm trainer python -m scripts.tuning.tune_baselines \
  policy=['baseline:astar_grid'] trials=50

# Обновление best‑params
docker compose run --rm trainer python -m scripts.tuning.update_best_params \
  policy=['baseline:astar_grid']
```

Пресеты скоринга: `scoring=default|fast|aggressive`.

## Best‑params

Если присутствует `configs/best_policy_params.json`, политики подхватывают параметры автоматически.
Отключить автозагрузку:
```bash
POLICY_BEST_PARAMS_DISABLE=1
```

Переопределить путь:
```bash
POLICY_BEST_PARAMS=/path/to/best_policy_params.json
```
