from __future__ import annotations

import json
from pathlib import Path

from ui.telemetry import TELEMETRY_SCHEMA_VERSION
from ui.telemetry_dto import telemetry_schema_json


def telemetry_schema_payload() -> dict[str, object]:
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "schema": telemetry_schema_json(),
    }


def write_schema(path: Path) -> Path:
    payload = telemetry_schema_payload()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
