"""Утилиты спецификации сцен для UI: редактирование, валидация, сериализация."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

try:
    import yaml
except Exception:
    yaml = None

try:
    from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
except Exception as exc:  # pragma: no cover - pydantic нужен для FastAPI
    raise RuntimeError("pydantic is required for ui.scene_spec") from exc

DEFAULT_FIELD_SIZE = 100.0
DEFAULT_MAX_STEPS = 600


class SceneValidationError(ValueError):
    pass


@dataclass
class SceneValidationResult:
    scene: dict[str, Any] | None
    errors: list[str]


class ThreatSpec(BaseModel):
    pos: tuple[float, float]
    radius: float = Field(gt=0)
    intensity: float = 0.1
    speed: float | None = None
    angle: float | None = None
    noise_scale: float | None = None
    vision_radius: float | None = None
    velocity: tuple[float, float] | None = None
    direction: tuple[float, float] | None = None

    model_config = ConfigDict(extra="allow")

    @field_validator("pos", "velocity", "direction", mode="before")
    @classmethod
    def _parse_point(cls, value: Any):
        if value is None:
            return value
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise ValueError("must be [x, y]")


class SceneSpec(BaseModel):
    id: str | None = None
    field_size: float = Field(default=DEFAULT_FIELD_SIZE, gt=0)
    max_steps: int = Field(default=DEFAULT_MAX_STEPS, ge=1)
    start_center: tuple[float, float] | None = None
    start_centers: list[tuple[float, float]] = Field(default_factory=list)
    start_sigma: float | None = None
    agents_pos: list[tuple[float, float]] = Field(default_factory=list)
    target_pos: tuple[float, float]
    walls: list[list[float]] = Field(default_factory=list)
    threats: list[ThreatSpec] = Field(default_factory=list)
    static_threats: list[ThreatSpec] = Field(default_factory=list)
    dynamic_threats: list[ThreatSpec] = Field(default_factory=list)

    model_config = ConfigDict(extra="allow")

    @field_validator("start_center", "target_pos", mode="before")
    @classmethod
    def _parse_point(cls, value: Any):
        if value is None:
            return value
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise ValueError("must be [x, y]")

    @field_validator("start_centers", "agents_pos", mode="before")
    @classmethod
    def _parse_point_list(cls, value: Any):
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("must be a list of [x, y]")
        out: list[tuple[float, float]] = []
        for idx, item in enumerate(value):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((float(item[0]), float(item[1])))
                continue
            raise ValueError(f"[{idx}] must be [x, y]")
        return out

    @field_validator("walls", mode="before")
    @classmethod
    def _parse_walls(cls, value: Any):
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("walls must be a list")
        out: list[list[float]] = []
        for idx, wall in enumerate(value):
            rect = None
            if isinstance(wall, (list, tuple)) and len(wall) == 4:
                rect = wall
            elif isinstance(wall, dict):
                if all(k in wall for k in ("x1", "y1", "x2", "y2")):
                    rect = [wall["x1"], wall["y1"], wall["x2"], wall["y2"]]
                elif all(k in wall for k in ("x", "y", "w", "h")):
                    rect = [
                        wall["x"],
                        wall["y"],
                        float(wall["x"]) + float(wall["w"]),
                        float(wall["y"]) + float(wall["h"]),
                    ]
            if rect is None:
                raise ValueError(f"walls[{idx}] must be [x1, y1, x2, y2] or dict")
            x1, y1, x2, y2 = (float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3]))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            out.append([x1, y1, x2, y2])
        return out

    @model_validator(mode="after")
    def _fill_defaults(self):
        if self.start_center is None:
            if not self.start_centers and not self.agents_pos:
                fs = float(self.field_size)
                self.start_center = (fs * 0.1, fs * 0.1)
        return self


def sanitize_scene_id(scene_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(scene_id or "").strip())
    cleaned = cleaned.strip("_")
    return cleaned or "custom_scene"


def _collect_errors(err: ValidationError) -> list[str]:
    out: list[str] = []
    for e in err.errors():
        loc = ".".join([str(x) for x in e.get("loc", [])])
        msg = e.get("msg", "invalid")
        if loc:
            out.append(f"{loc}: {msg}")
        else:
            out.append(str(msg))
    return out


def _bounds_check(scene: SceneSpec) -> list[str]:
    errors: list[str] = []
    fs = float(scene.field_size)

    def _point_out_of_bounds(point: tuple[float, float]) -> bool:
        x, y = point
        return x < 0 or y < 0 or x > fs or y > fs

    if scene.start_center is not None:
        if _point_out_of_bounds(scene.start_center):
            errors.append("start_center out of bounds")
    for idx, point in enumerate(scene.start_centers or []):
        if _point_out_of_bounds(point):
            errors.append(f"start_centers[{idx}] out of bounds")
    for idx, point in enumerate(scene.agents_pos or []):
        if _point_out_of_bounds(point):
            errors.append(f"agents_pos[{idx}] out of bounds")
    if scene.target_pos is not None:
        if _point_out_of_bounds(scene.target_pos):
            errors.append("target_pos out of bounds")
    for idx, w in enumerate(scene.walls or []):
        x1, y1, x2, y2 = w
        if x1 < 0 or y1 < 0 or x2 > fs or y2 > fs:
            errors.append(f"walls[{idx}] out of bounds")

    def _check_threats(name: str, items: list[ThreatSpec]) -> None:
        for idx, t in enumerate(items or []):
            if _point_out_of_bounds(t.pos):
                errors.append(f"{name}[{idx}].pos out of bounds")
            if t.radius <= 0:
                errors.append(f"{name}[{idx}].radius must be > 0")

    _check_threats("threats", scene.threats)
    _check_threats("static_threats", scene.static_threats)
    _check_threats("dynamic_threats", scene.dynamic_threats)
    return errors


def _raw_bounds_errors(scene: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        fs = float(scene.get("field_size", DEFAULT_FIELD_SIZE))
    except Exception:
        return errors
    target = scene.get("target_pos")
    if isinstance(target, (list, tuple)) and len(target) == 2:
        try:
            x = float(target[0])
            y = float(target[1])
        except Exception:
            return errors
        if x < 0 or y < 0 or x > fs or y > fs:
            errors.append("target_pos out of bounds")

    def _raw_point_errors(name: str, value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, list):
            return
        points = value if name.endswith("s") else [value]
        for idx, point in enumerate(points):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                x = float(point[0])
                y = float(point[1])
            except Exception:
                continue
            if x < 0 or y < 0 or x > fs or y > fs:
                errors.append(f"{name}[{idx}] out of bounds")

    _raw_point_errors("start_centers", scene.get("start_centers"))
    _raw_point_errors("agents_pos", scene.get("agents_pos"))

    def _raw_threat_errors(name: str, value: Any) -> None:
        if not isinstance(value, list):
            return
        for idx, raw in enumerate(value):
            if not isinstance(raw, dict):
                continue
            pos = raw.get("pos")
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                try:
                    x = float(pos[0])
                    y = float(pos[1])
                except Exception:
                    continue
                if x < 0 or y < 0 or x > fs or y > fs:
                    errors.append(f"{name}[{idx}].pos out of bounds")

    _raw_threat_errors("threats", scene.get("threats"))
    _raw_threat_errors("static_threats", scene.get("static_threats"))
    _raw_threat_errors("dynamic_threats", scene.get("dynamic_threats"))
    return errors


def validate_and_normalize(scene: dict[str, Any] | None, *, allow_missing_id: bool = False) -> SceneValidationResult:
    if not isinstance(scene, dict):
        return SceneValidationResult(None, ["scene must be a dict"])
    errors: list[str] = []
    try:
        model = SceneSpec.model_validate(scene)
    except ValidationError as exc:
        errors = _collect_errors(exc)
        if not allow_missing_id and not scene.get("id"):
            errors.append("id is required")
        errors.extend(_raw_bounds_errors(scene))
        return SceneValidationResult(None, errors)

    if not allow_missing_id:
        if not model.id:
            errors.append("id is required")
    if model.id is not None:
        model.id = str(model.id)

    errors.extend(_bounds_check(model))
    if errors:
        return SceneValidationResult(None, errors)

    data = model.model_dump(exclude_none=True)
    for optional_empty_key in ("start_centers", "agents_pos", "static_threats", "dynamic_threats"):
        if optional_empty_key not in scene and data.get(optional_empty_key) == []:
            data.pop(optional_empty_key, None)

    def _to_jsonable(value: Any):
        if isinstance(value, tuple):
            return [_to_jsonable(v) for v in value]
        if isinstance(value, list):
            return [_to_jsonable(v) for v in value]
        if isinstance(value, dict):
            return {k: _to_jsonable(v) for k, v in value.items()}
        return value

    return SceneValidationResult(_to_jsonable(data), [])


def parse_scene_text(text: str) -> dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise SceneValidationError("scene text is empty")
    data = None
    try:
        data = json.loads(raw)
    except Exception as err:
        if yaml is None:
            raise SceneValidationError("YAML parser is not available") from err
        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise SceneValidationError("scene text must decode to a dict")
    return data


def export_scene_text(scene: dict[str, Any], fmt: str = "json") -> str:
    fmt = str(fmt or "json").lower()
    if fmt == "json":
        return json.dumps(scene, ensure_ascii=False, indent=2)
    if fmt in {"yaml", "yml"}:
        if yaml is None:
            raise SceneValidationError("YAML parser is not available")
        return yaml.safe_dump(scene, sort_keys=False, allow_unicode=True)
    raise SceneValidationError(f"unknown format: {fmt}")
