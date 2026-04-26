"""Контракт телеметрии: версия схемы и payload."""

from ui.telemetry import TELEMETRY_SCHEMA_VERSION
from ui.telemetry_schema import telemetry_schema_payload


def test_telemetry_schema_payload_version():
    payload = telemetry_schema_payload()
    assert payload.get("schema_version") == TELEMETRY_SCHEMA_VERSION
    assert "schema" in payload
