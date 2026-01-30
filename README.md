# Threat-aware Swarm — MARL-симулятор роя БПЛА с динамическими угрозами

Этот репозиторий — пет-проект по multi-agent reinforcement learning (MARL).
Идея проекта в следующем: рой дронов учится долетать до цели, выживать и вести себя как рой (не слипаться в одну вереницу), когда на поле есть зоны угроз.

Стек: PettingZoo (ParallelEnv) + SuperSuit + Stable-Baselines3 (PPO), с parameter sharing (одна политика на всех агентов).

---

## Что умеет проект

- 2D-среда с границами поля и статическими (в будущем динамическими) угрозами (у каждой свой радиус + шанс смерти при нахождении внутри в тик).
- Рой из `N` агентов (дронов) с общей политикой (PPO).
- Награда учитывает:
  - прогресс к цели;
  - удержание в зоне цели (с дожатием в центр);
  - риск поражения угрозами;
  - штраф за смерть;
  - штраф за близость к соседям (с отключением в зоне цели, чтобы финальный сбор не карался).
- Логи в TensorBoard: train/rollout метрики PPO + дополнительные swarm-метрики (alive/finished/in_goal/distance и т.п.).
- Скрипты для:
  - обучения (`scripts/trained_ppo.py`);
  - индексации моделей и прогона оценки по реестру (`scripts/index_models.py`, `scripts/eval_models.py`);
  - визуализации поведения (`viz/run_trained_pz.py`, `viz/run_demo.py`).

---

## Установка

Рекомендуемый Python: **3.10+**.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Если запускаете скрипты из корня репозитория, удобно добавлять `PYTHONPATH=.`:

```bash
PYTHONPATH=. python -c "import env; print('ok')"
```

---

## Быстрый старт

### Обучение PPO (основной способ)

Главный скрипт обучения — `scripts/trained_ppo.py`.

Пример (команда из корня репо):

```bash
PYTHONPATH=. python scripts/trained_ppo.py \
  --run_dir runs/run_$(date +%Y%m%d_%H%M%S) \
  --timesteps 6000000 \
  --save_every 2000000 \
  --eval_every 20000 \
  --n_eval_episodes 20 \
  --max_steps 600
```

Что получится:
- `runs/<run_id>/models/` — чекпоинты и `best_by_finished.zip`
- `runs/<run_id>/tb/` — TensorBoard логи
- `runs/<run_id>/meta/run.json` — метаданные запуска

TensorBoard:

```bash
tensorboard --logdir runs
```

### 3) Визуализация обученной модели

Обычно запускаю так:

```bash
PYTHONPATH=. python -m viz.run_trained_pz
```

Внутри `viz/run_trained_pz.py` выбирается модель (по умолчанию — `best_by_finished` из последнего прогона).
Внутри скрипта есть аргумент `--model`, можно указать путь к конкретной модели прямо в командной строке:

```bash
PYTHONPATH=. python -m viz.run_trained_pz --model runs/run_20260118_203327/models/best_by_finished.zip
```

---

## Как сравниваются модели

1) Сформировать реестр:

```bash
PYTHONPATH=. python scripts/index_models.py --scan runs --out model_registry.csv --rewrite
```

2) Прогнать оценку (например, 20 эпизодов, фиксированный и случайный seed):

```bash
PYTHONPATH=. python scripts/eval_models.py --registry model_registry.csv --mode both --n-episodes 20 --max-steps 600 --deterministic --seed 0
```

В `model_registry.csv` добавятся колонки с результатами и ошибками (если какие-то модели несовместимы).

---

## Заметки по воспроизводимости

В проекте специально хранятся метаданные каждого прогона (`runs/<id>/meta/run.json`), потому что по одному TensorBoard потом тяжело восстановить:
- какими были гиперпараметры;
- какая версия среды была;
- какие флаги были включены.

Notebook’ (`notebooks/`) используется как лаборатория: там графики, запуски прогонов.
Скриптом для запуска обучения является `scripts/trained_ppo.py`.
