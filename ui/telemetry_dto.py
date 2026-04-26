"""Типизированные DTO для телеметрии UI."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

try:
    from pydantic import ConfigDict
except Exception:  # pragma: no cover - совместимость
    ConfigDict = None


class _BaseTelemetry(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:  # pragma: no cover - Pydantic v1

        class Config:
            extra = "forbid"


class TelemetryStats(_BaseTelemetry):
    step: int
    alive: int | None
    finished: int | None
    in_goal: int | None
    mean_dist: float | None
    mean_risk: float | None
    mean_path_ratio: float | None
    mean_threat_collisions: float | None
    mean_energy: float | None
    mean_energy_level: float | None


class TelemetryAgent(_BaseTelemetry):
    id: str
    index: int
    pos: list[float]
    vel: list[float]
    alive: bool
    finished: bool
    in_goal: bool
    dist: float | None
    risk_p: float | None
    path_ratio: float | None
    collided: bool
    threat_collided: bool
    min_dist_to_threat: float | None
    energy: float | None
    energy_level: float | None
    action: list[float] | None


class TelemetryThreat(_BaseTelemetry):
    pos: list[float]
    radius: float
    intensity: float
    kind: str
    dynamic: bool
    oracle_block: bool


class TelemetryPayload(_BaseTelemetry):
    schema_version: str
    stats: TelemetryStats
    agents: list[TelemetryAgent]
    threats: list[TelemetryThreat]
    walls: list[list[float]]
    oracle_path: list[list[float]]
    field_size: float
    goal_radius: float
    target_pos: list[float] | None
    screen_size: int
    grid_res: float
    agent_index: int
    agent_id: str | None
    agent_grid: list[list[float]] | None
    agent_obs: dict[str, Any] | None
    wind: list[float] | None

    def to_dict(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()  # pragma: no cover - Pydantic v1


def telemetry_schema_json() -> dict[str, Any]:
    if hasattr(TelemetryPayload, "model_json_schema"):
        return TelemetryPayload.model_json_schema()
    return TelemetryPayload.schema()  # pragma: no cover - Pydantic v1
