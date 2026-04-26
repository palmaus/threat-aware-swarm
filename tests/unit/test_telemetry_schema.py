from __future__ import annotations

import json
from pathlib import Path

from ui.telemetry_schema import write_schema


def test_telemetry_schema_written(tmp_path: Path) -> None:
    out = tmp_path / "schema.json"
    write_schema(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "schema_version" in data
    assert "schema" in data
    assert "properties" in data["schema"]
