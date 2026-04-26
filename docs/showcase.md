# Showcase

Это короткая презентационная страница репозитория. Её стоит открывать, когда нужно быстро объяснить проект ревьюеру, рекрутеру или другому инженеру.

## Задача

Проект исследует мультиагентную навигацию в 2D-мире с:
- частичной наблюдаемостью;
- динамическими угрозами;
- непрерывным управлением;
- компромиссом между безопасностью и эффективностью.

Это сознательно сложнее, чем игрушечный gridworld. Агенты не прыгают по клеткам, а движутся через waypoint-управление с drag, wind и ограничениями на ускорение.

## Что построено

- собственная swarm-среда на PettingZoo;
- сильные классические бейзлайны: `A*`, `MPC-lite`, `Potential Fields`, `Flow Field`;
- стек PPO / RecurrentPPO с `AdvancedSwarmCNN`;
- React UI в стиле Mission Control для телеметрии, compare mode, replay и scene editor;
- benchmark-система с разделением на fair / privileged / oracle-only;
- tuning pipeline с holdout-валидацией, OOD-проверками и perf-aware tooling.

## Почему этот репозиторий сильнее типичного RL pet-проекта

- здесь evaluation — часть продукта, а не постфактум;
- learned policy сравнивается с осмысленными планировщиками;
- режимы доступа к информации формализованы явно;
- есть воспроизводимый runtime, tracking, отчёты и demo tooling;
- есть рабочий UI, а не только статические графики из ноутбука.

## Что открыть в первую очередь

1. `README.md` — обзор проекта и быстрый старт.
2. `docs/benchmark_policy.md` — правила честного сравнения.
3. `docs/perf/README.md` — fixed-step perf protocol.
4. `docs/infra/ui.md` — архитектура UI и telemetry contract.

## Что запустить для быстрой проверки

```bash
make warm
docker compose up -d web mlflow
docker compose run --rm trainer python -m scripts.eval.eval_scenarios \
  policy=baseline:potential_fields \
  scenes=[preset:sanity] \
  episodes=2
```

## Артефакты для витрины и релиза

- benchmark release bundle: `python3 -m scripts.analysis.build_benchmark_release ...`
- demo/report pack: `make demo-pack && make demo-report && make demo-story`
- fixed-step perf profile: `make profile-baselines` / `make perf-gate`
