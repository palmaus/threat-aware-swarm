"""Диспетчер для валидированных управляющих WebSocket‑сообщений."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ui.ws_models import ControlBase, parse_control_message

Handler = Callable[[Any, dict[str, Any]], list[dict[str, Any]]]


class ControlDispatcher:
    def __init__(self, handlers: dict[str, Handler]) -> None:
        self._handlers = handlers

    def dispatch(self, session: Any, raw: dict[str, Any]) -> list[dict[str, Any]]:
        msg, err = parse_control_message(raw)
        if msg is None:
            if err:
                session.last_error = err
            return []
        return self._dispatch_validated(session, msg)

    def _dispatch_validated(self, session: Any, msg: ControlBase) -> list[dict[str, Any]]:
        handler = self._handlers.get(msg.action)
        if handler is None:
            session.last_error = f"Unknown handler for action: {msg.action}"
            return []
        payload = msg.model_dump(by_alias=True)
        return handler(session, payload)
