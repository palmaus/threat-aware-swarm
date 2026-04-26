# `common/` — общие контракты и runtime-утилиты

Эта папка хранит только кросс-срезовые сущности, которые используются в
нескольких подсистемах проекта одновременно.

## Текущая таксономия

### `common/policy`
- `context.py` — единый `PolicyContext` и адаптеры `context_from_state` /
  `context_from_payload`
- `oracle_visibility.py` — правила видимости oracle для baseline/RL policy
- `obs_schema.py` — единый layout `obs@1694:v5` для runtime/UI/baseline
- `specs.py` — lightweight baseline capability/spec registry, доступный до
  импорта concrete baseline modules
- `waypoint_controller.py` — общий waypoint-to-control adapter без зависимости
  `env -> baselines`

### `common/runtime`
- `contracts.py` — runtime guardrails для `SimState`, `PublicState` и reset-observation contract
- `episode_runner.py` — общий dispatch policy actions/context для eval/UI/bench
- `env_factory.py`, `env_overrides.py` — общий слой создания `SwarmPZEnv`/VecEnv
  и разделения structured `EnvConfig` полей от runtime-only override'ов
- `env_factory.py` лениво импортирует `supersuit` и fail-close на ошибках
  runtime curriculum/stage apply, чтобы UI/runtime path не зависел от vec stack
  и не возвращал partially-configured env
- `path_utils.py`, `hydra_utils.py`, `xai.py` — runtime helpers, которые могут
  использоваться не только script-entrypoint'ами

### `common/physics`
- `model.py` — shared physics helper'ы для low-level control и env runtime
- `walls.py` — public wall-collision helpers для physics/planner'ов
- `walls_numba.py` — optional Numba kernels с той же wall-collision семантикой,
  чтобы planner'ы не держали локальные копии physics-ядра

## Совместимость

Старые плоские пути импорта пока оставлены:
- `common.context`
- `common.contracts`
- `common.oracle_visibility`
- `common.physics_model`

Они работают как compatibility layer и просто переэкспортируют новые
подпакеты. Для нового кода предпочтительны именно вложенные пути.

## Правило

Если модуль нужен только одной подсистеме, его не надо класть в `common/`.
`common/` — это не склад утилит, а место для реально общих контрактов.
