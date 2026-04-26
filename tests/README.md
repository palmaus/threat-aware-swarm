# tests/

Тестовый контур проекта: unit, integration, smoke‑прогоны.

## Запуск

```bash
# полный прогон
python3 -m pytest

# быстрый unit‑контур
pytest tests/unit -q

# конкретный файл
pytest tests/unit/test_pz_env_api.py -q
```

## Советы

- При отладке контракта включай строгие проверки:
  `TA_STRICT_DEBUG=1 pytest tests/unit -q`
- Если тесты нестабильны — проверь seed‑ы и сценарии.
