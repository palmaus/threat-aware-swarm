# Benchmark Policy

Этот документ фиксирует правила честного сравнения методов. Его задача — не дать смешать fair, privileged и oracle-only результаты в одну таблицу.

См. также: `docs/benchmarks/leaderboards.md`, `docs/benchmarks/metric_semantics.md`.

## Базовые принципы

1. `Oracle` — это reference/eval entity, а не участник main leaderboard.
2. Main benchmark строится только на сопоставимом доступе к информации.
3. Privileged методы допустимы, но только в отдельной таблице.
4. `Flow Field` — planner/reference, а не oracle.
5. На динамике static oracle — это lower bound по длине пути, а не истинный optimum.

## Режимы доступа к информации

| Режим | Что разрешено | Типичные методы |
| --- | --- | --- |
| `fair` / `causal` | только локальные наблюдения и разрешённая память | PPO, Potential Fields, MPC-lite, A* memory, discovered-map planners |
| `privileged` / `map-aware` | полная статическая карта заранее | full-map A*, full-map Flow Field |
| `oracle-only` | любые недостижимые для policy сигналы, но только для eval | static shortest path, distance-map lower bound |

## Политика для main benchmark

### Что запрещено
- `oracle_visibility != none`;
- oracle distance/action/waypoint как вход в policy;
- future threat positions или скрытое simulator state;
- full static map для методов, заявленных как fair.

### Что разрешено
- считать `path_ratio` и другие нормирующие метрики через oracle в eval;
- использовать discovered memory/map, если она строится только из реально наблюдаемой информации;
- публиковать privileged и oracle reference отдельно от fair таблицы.

### Практический профиль
Для main benchmark используй:

```bash
env.oracle_visibility=none
env.oracle_visible_to_baselines=false
env.oracle_visible_to_agents=false
# или
+oracle=fair
```

## Группы экспериментов

### Group A — Main fair benchmark
Главная таблица проекта. Здесь сравниваются только методы с сопоставимым доступом к информации.

### Group B — Privileged references
Показывает, что даёт доступ к полной статической карте. Это не main leaderboard.

### Group C — Oracle reference block
Нужен для нормировки и интерпретации (`path_ratio`, lower bounds), но не для ранжирования policy.

## Статус Flow Field

| Вариант | Категория | Где публиковать |
| --- | --- | --- |
| `Flow Field` по полной карте | privileged reference | privileged tables |
| `Flow Field` по discovered map | fair baseline | main fair benchmark, если режим действительно честный |
| `Flow Field + local dynamic avoidance` | hybrid baseline | fair или privileged в зависимости от источника карты |

Ключевое правило: на динамике `Flow Field` не считается upper bound. Это сильный planner с static prior, а не теоретический optimum.

## Checklist перед публикацией результатов
- У метода явно указан `information_regime`.
- Oracle не используется как вход в policy для fair benchmark.
- Dynamic и static результаты не смешаны в один scoreboard.
- Для privileged методов есть отдельная таблица.
- Для oracle reference не используется язык вида “baseline winner”.

## Связанные артефакты
- Форматы таблиц: `docs/benchmarks/leaderboards.md`
- Трактовка метрик: `docs/benchmarks/metric_semantics.md`
- Leakage SSoT: `configs/benchmark/leakage_checklist.yaml`
