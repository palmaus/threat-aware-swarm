# Metric Semantics

Этот документ задаёт единый смысл основных метрик. Он нужен, чтобы leaderboard, tuning reports и demo-сравнения не расходились по трактовке.

## Общие правила
- `Static` и `Dynamic` не смешиваются в одну таблицу.
- `Best run` и aggregated benchmark — это разные сущности.
- Oracle используется только для нормировки и eval, но не как вход в fair policy.

## Базовая нотация

Пусть:
- `E` — число эпизодов;
- `A` — число агентов;
- `T_e` — длина эпизода `e`;
- `finished_e` — число финишировавших агентов;
- `alive_e` — число выживших агентов в конце;
- `risk_p[a,t]` — риск агента `a` на шаге `t`;
- `path_len[a]` — длина фактического пути агента;
- `optimal_len[a]` — oracle-длина пути по статике;
- `action_t[a]` — действие агента `a` на шаге `t`.

## Основные метрики

| Метрика | Формула / определение | Что означает |
| --- | --- | --- |
| `success_rate` | `mean_e(1[finished_frac_end_e >= threshold])` | доля успешных эпизодов |
| `finished_frac_end` | `finished_e / A` | доля агентов, достигших цели |
| `alive_frac_end` | `alive_e / A` | доля живых агентов в конце |
| `deaths_mean` | `mean_e(A - alive_e)` | среднее число погибших |
| `risk_integral_all` | `(1 / T_e) * sum_t mean_a(risk_p[a,t])` | средний риск по всем агентам |
| `risk_integral_alive` | `(1 / T_e) * sum_t mean_{alive}(risk_p[a,t])` | риск среди живых |
| `time_to_goal_mean` | `mean_{finished}(decision_finish_step[a])`, где первый `env.step()` = 1 | среднее время до финиша |
| `collision_like` | `mean_a(1[min_neighbor_dist[a] < sep_radius])` | прокси столкновений |
| `episode_len_mean` | `mean_e(T_e)` | средняя длина эпизода |
| `path_ratio` | `mean_a(path_len[a] / optimal_len[a])` | эффективность пути против static oracle |
| `action_smoothness` | `(1 / T_e) * sum_t mean_a(||action_t[a] - action_{t-1}[a]||)` | плавность управления |
| `energy_efficiency` | `path_len_mean / energy_spent` | путь на единицу энергии |
| `cost_*` | `(1 / T_e) * sum_t mean_a(cost_*[a,t])` | компоненты cost decomposition |

## Как читать ключевые метрики

### Success / finish / alive
- `success_rate` — operational metric первого порядка.
- `finished_frac_end` полезна для swarm-задачи, где успех эпизода не обязательно бинарен.
- `alive_frac_end` — safety envelope, особенно важна на dynamic threats.

### Risk и safety
- `risk_integral_*` показывает цену движения через опасные зоны.
- `collision_like` — приближённая safety-метрика, не заменяет полноценный crash accounting.
- `cost_*` полезны только если фиксирована одна и та же схема reward/cost.

### Return
`Mean return` полезен внутри одного reward design. Между разными reward-схемами его нельзя считать универсальной метрикой качества.

### Path ratio

```text
path_ratio = actual_path_length / oracle_static_shortest_path_length
```

- На `static` это приближённая мера геометрической эффективности.
- На `dynamic` это не gap к истинному optimum. Это нижняя граница по статике и одновременно цена безопасности, реактивности и неполного знания.

### Latency
Latency обязательна для planner-based baseline'ов. Без неё сравнение A*/MPC/Flow Field с PPO неполно.

## Агрегация
- внутри сцены: агрегируй по эпизодам;
- между сценами: используй `macro-average`, а не “свалить все эпизоды в одну кучу”;
- для main benchmark публикуй как минимум `mean/std`, а лучше `mean/std/95% CI`.

## Нормализация для сравнительных графиков

Если на одном графике сравниваются разнонаправленные метрики:
- `higher is better`: `value / baseline`;
- `lower is better`: `baseline / value`.

Всегда подписывай, что нормализация учитывает направление качества. Иначе график легко читать неверно.

## Что считать хорошим отчётом
- Есть явное разделение `static` и `dynamic`.
- Указано, какие метрики “higher is better”, а какие — “lower is better”.
- `path_ratio` на dynamic сопровождается пояснением про lower bound.
- Demo best-run не подаётся как replacement для aggregated benchmark.
