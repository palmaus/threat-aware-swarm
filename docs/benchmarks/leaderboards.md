# Leaderboards & Result Tables

Этот документ задаёт единый шаблон публикации результатов. Его смысл — чтобы README, demo-report и benchmark-отчёты использовали одну и ту же структуру.

См. также: `docs/benchmark_policy.md`, `docs/benchmarks/metric_semantics.md`.

## Правила публикации
- `Static` и `Dynamic` публикуются раздельно.
- Main leaderboard содержит только `fair` методы.
- Privileged методы и oracle references выносятся в отдельные блоки.
- Demo best-run не подменяет aggregated benchmark.

## Таблица 1 — Main Fair Benchmark (Static)

| Method | Info regime | Scenario family | Success | Collision | Risk cost | Mean return | Mean path length | Path ratio | Mean time to goal | Latency | Seeds | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| PPO | local_only | static | — | — | — | — | — | — | — | — | — | — |
| Potential Fields | local_only | static | — | — | — | — | — | — | — | — | — | — |
| MPC-lite | local_only | static | — | — | — | — | — | — | — | — | — | — |
| A* Grid (memory) | local_memory | static | — | — | — | — | — | — | — | — | — | — |

## Таблица 2 — Main Fair Benchmark (Dynamic)

| Method | Info regime | Scenario family | Success | Collision | Risk cost | Mean return | Mean path length | Path ratio | Mean time to goal | Latency | Seeds | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| PPO | local_only | dynamic | — | — | — | — | — | — | — | — | — | — |
| Potential Fields | local_only | dynamic | — | — | — | — | — | — | — | — | — | — |
| MPC-lite | local_only | dynamic | — | — | — | — | — | — | — | — | — | — |
| A* Grid (memory) | local_memory | dynamic | — | — | — | — | — | — | — | — | — | — |

> `Flow Field` попадает в fair таблицу только если используется discovered-map версия без privileged knowledge.

## Таблица 3 — Privileged Map-Aware References

| Method | Info regime | Scenario family | Success | Collision | Risk cost | Mean path length | Path ratio | Latency | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Flow Field (full map) | full_static_map | static | — | — | — | — | — | — | — |
| A* (full map) | full_static_map | static | — | — | — | — | — | — | — |
| Flow Field + local avoidance | full_static_map | dynamic | — | — | — | — | — | — | — |

## Таблица 4 — Oracle Reference Block

| Scenario family | Oracle path length | Availability during inference | Usage | Notes |
| --- | ---: | --- | --- | --- |
| static | — | no | lower bound for `path_ratio` | — |
| dynamic | — | no | reference only | not an upper bound |

## Таблица 5 — Robustness / OOD Summary

| Method bucket | ID success | OOD success | Delta success | ID path ratio | OOD path ratio | Delta safety cost | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Best PPO fair | — | — | — | — | — | — | — |
| Best fair heuristic | — | — | — | — | — | — | — |
| Best privileged reference | — | — | — | — | — | — | — |

## Таблица 6 — Leakage Checklist

Source of truth: `configs/benchmark/leakage_checklist.yaml`.

| Method | Uses full static map | Uses oracle_dir | Uses future threats | Uses hidden state | Privileged train signal | Privileged inference signal | Allowed in main benchmark |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | no | no | no | no | no | no | yes |
| Potential Fields | no | conditional | no | no | no | conditional | yes, only with fair oracle settings |
| MPC-lite | no | conditional | no | no | no | conditional | yes, only with fair oracle settings |
| A* Grid (memory) | no | conditional | no | no | no | conditional | yes, only with fair oracle settings |
| Flow Field | yes | yes | no | no | no | yes | no |
| Oracle | yes | yes | yes | yes | eval-only | eval-only | no |

## Минимальный reporting checklist
- В каждой строке есть `information_regime`.
- Есть отдельная таблица для `privileged` результатов.
- Есть отдельный reference block для oracle.
- Указаны `Seeds` и хотя бы `Latency` для planner-based методов.
- Для dynamic таблиц `path_ratio` явно трактуется как lower-bound comparison, а не true optimum gap.
