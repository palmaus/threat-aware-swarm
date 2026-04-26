# env/

Пакет среды теперь разложен по подсистемам, а не по длинному плоскому списку файлов.

## Таксономия

| Подсистема | Пакет | Что лежит внутри |
| --- | --- | --- |
| Runtime | `env/` | `pz_env.py`, `engine.py`, `decision_loop.py`, `events.py`, `event_handlers.py`, `state.py`, `config.py` |
| Observations | `env/observations/` | builder и observer для `Dict`-obs |
| Oracles | `env/oracles/` | distance field, path oracle, oracle manager |
| Rewards | `env/rewards/` | reward pipeline, DTO, cost schema, metrics, rewarder |
| Physics | `env/physics/` | force-based physics core, physics loop, wind |
| Scenes | `env/scenes/` | threat models, procedural generation, providers, scene manager, spawn controller |
| Shared utils | `env/utils/` | geometry helpers |

## Архитектурная идея

- `SwarmPZEnv` остаётся тонким PettingZoo-adapter.
- `SwarmEngine` держит runtime orchestration и step/reset lifecycle.
- `SwarmEngine` кэширует `PublicState` на тик и держит runtime/export payload раздельно.
- Подсистемы теперь собраны по ответственности:
  - наблюдения отдельно,
  - награды и cost-схема отдельно,
  - физика отдельно,
  - сцены и угрозы отдельно,
  - oracle отдельно.

Это уменьшает ощущение «склада файлов» и упрощает точечные изменения без случайного затрагивания соседних областей.

## Наблюдения

- формат: `Dict`-obs only;
- версия: `obs@1694:v5`;
- блоки: `to_target`, `vel`, `walls`, `last_action`, `measured_accel`, `energy_level`, `grid (41×41)`.

Схема: `docs/reference/env_schema.md`.

## Полезные проверки

```bash
# Env / wrapper sanity
docker compose run --rm trainer python -m scripts.debug.debug_env_metrics steps=2000

# Goal-hold / finish behavior
docker compose run --rm trainer python -m scripts.debug.finish_debug policy=baseline:potential_fields episodes=10
```

## Env profiles

- `env_profile=lite` — облегчённая среда (быстрый runtime)
- `env_profile=full` — полный physics profile
- `env.runtime_mode=full` — полный export/debug/runtime контур
- `env.runtime_mode=train_fast` — урезанный runtime для train/eval-lite: меньше debug/info/oracle payload

```bash
docker compose run --rm trainer python -m scripts.train.trained_ppo \
  experiment=ppo_waypoint env_profile=lite env.runtime_mode=train_fast
```
