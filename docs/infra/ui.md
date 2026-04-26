# Web UI

Web UI — это React-based Mission Control интерфейс для симуляции, сравнения политик и интерактивной отладки.

## Архитектурная схема
- backend: `ui.web_server` + FastAPI/WebSocket;
- frontend: `ui_frontend/` (Vite + React + TypeScript + Zustand);
- transport: binary/msgpack telemetry;
- rendering: клиентский Canvas, без серверного Pygame render loop.

## Быстрый старт

### Запуск через compose
```bash
docker compose up web
# http://127.0.0.1:8000
```

### Локальная пересборка frontend bundle
```bash
make ui-build
```

### Smoke test
```bash
make ui-smoke
```

## Что умеет UI
- single-session и compare mode;
- demo / research представления;
- timeline/time-travel replay;
- agent inspector, charts, minimap, scene editor;
- oracle toggles, attention heatmaps, GIF export;
- telemetry schema validation на клиенте.

## Контракты

| Контракт | Где зафиксирован |
| --- | --- |
| Web config | `configs/hydra/ui/web.yaml` |
| Telemetry schema | `ui_frontend/dist/telemetry_schema.json` |
| DTO / validation | `ui/dispatcher.py`, `ui/scene_spec.py` |

Текущий runtime-контур — только React. Legacy UI удалён из основного пути.

## Важные правила
- если меняется `TelemetryPayload`, обновляй schema generation и UI consumer одновременно;
- если меняется control message, обновляй Pydantic model и React-side action sender вместе;
- любые тяжёлые smoke-тесты UI должны оставаться отделёнными от обычного быстрого pytest-контура.

## Частые проблемы

### Пустой canvas
Проверь, что фронтенд собран и актуален:
```bash
make ui-build
```

### Несовпадение версии telemetry schema
UI должен показывать ошибку schema mismatch. В этом случае пересобери bundle и убедись, что backend и frontend взяты из одного состояния кода.

### Лаги или зависания compare mode
Сначала проверь тяжёлые baseline'ы (`mpc_lite`, `astar_grid`) и количество `policy_workers`.

## Связанные документы
- `AGENTS.md` — общий runtime-контекст
- `docs/infra/docker.md` — запуск и сервисы
- `docs/logs/research_log.md` — история UI миграции и фиксов
