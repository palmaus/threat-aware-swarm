"""Pydantic‑модели для управляющих WebSocket‑сообщений."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class ControlBase(BaseModel):
    type: Literal["control"] = "control"
    action: str

    model_config = ConfigDict(extra="ignore")


class PauseMessage(ControlBase):
    action: Literal["pause"]


class StepMessage(ControlBase):
    action: Literal["step"]


class ResetMessage(ControlBase):
    action: Literal["reset"]


class NewMapMessage(ControlBase):
    action: Literal["new_map"]


class CompareMessage(ControlBase):
    action: Literal["compare"]
    value: bool


class PolicyMessage(ControlBase):
    action: Literal["policy"]
    name: str
    side: Literal["left", "right"] | None = None


class SceneMessage(ControlBase):
    action: Literal["scene"]
    name: str


class SeedMessage(ControlBase):
    action: Literal["seed"]
    value: int


class FpsMessage(ControlBase):
    action: Literal["fps"]
    value: int


class AgentMessage(ControlBase):
    action: Literal["agent"]
    value: int


class ModelMessage(ControlBase):
    action: Literal["model"]
    value: str
    deterministic: bool | None = None
    side: Literal["left", "right"] | None = None


class DeterministicMessage(ControlBase):
    action: Literal["deterministic"]
    value: bool
    side: Literal["left", "right"] | None = None


class ToggleMessage(ControlBase):
    action: Literal["toggle"]
    name: str
    value: bool


class AttentionChannelMessage(ControlBase):
    action: Literal["attention_channel"]
    value: Literal["sum", "x", "y"] | str = "sum"


class OracleMessage(ControlBase):
    action: Literal["oracle"]
    enabled: bool | None = None
    async_mode: bool | None = Field(default=None, alias="async")
    interval: int | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TuneMessage(ControlBase):
    action: Literal["tune"]
    params: dict[str, Any] = Field(default_factory=dict)


class ScenePreviewMessage(ControlBase):
    action: Literal["scene_preview"]
    scene: dict[str, Any]


class SceneSaveMessage(ControlBase):
    action: Literal["scene_save"]
    scene: dict[str, Any]


class SceneDeleteMessage(ControlBase):
    action: Literal["scene_delete"]
    scene_id: str


class SceneRefreshMessage(ControlBase):
    action: Literal["scene_refresh"]


class SceneParseMessage(ControlBase):
    action: Literal["scene_parse"]
    text: str


class SceneExportMessage(ControlBase):
    action: Literal["scene_export"]
    scene: dict[str, Any]
    format: str = "json"


ACTION_MODELS: dict[str, type[ControlBase]] = {
    "pause": PauseMessage,
    "step": StepMessage,
    "reset": ResetMessage,
    "new_map": NewMapMessage,
    "compare": CompareMessage,
    "policy": PolicyMessage,
    "scene": SceneMessage,
    "seed": SeedMessage,
    "fps": FpsMessage,
    "agent": AgentMessage,
    "model": ModelMessage,
    "deterministic": DeterministicMessage,
    "toggle": ToggleMessage,
    "attention_channel": AttentionChannelMessage,
    "oracle": OracleMessage,
    "tune": TuneMessage,
    "scene_preview": ScenePreviewMessage,
    "scene_save": SceneSaveMessage,
    "scene_delete": SceneDeleteMessage,
    "scene_refresh": SceneRefreshMessage,
    "scene_parse": SceneParseMessage,
    "scene_export": SceneExportMessage,
}


def parse_control_message(raw: dict[str, Any]) -> tuple[ControlBase | None, str | None]:
    action = raw.get("action") if isinstance(raw, dict) else None
    if not action or not isinstance(action, str):
        return None, "Missing or invalid action in control message."
    model_cls = ACTION_MODELS.get(action)
    if model_cls is None:
        return None, f"Unknown control action: {action}."
    try:
        return model_cls.model_validate(raw), None
    except ValidationError as exc:
        return None, str(exc)
