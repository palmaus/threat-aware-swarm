"""FastAPI‑приложение для интерактивного веб‑интерфейса."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

# В UI отключаем Numba‑JIT по умолчанию: снижает риск нативных крашей в длительных сессиях.
if os.environ.get("TA_DISABLE_NUMBA_UI", "1") != "0":
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import hydra
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from omegaconf import OmegaConf

try:
    import msgpack
except Exception:  # pragma: no cover - запасной путь для минимальных установок
    msgpack = None
from common.runtime.hydra_utils import apply_schema
from common.runtime.path_utils import resolve_repo_path
from ui.config import UIConfig, WebUIConfig
from ui.controller import SwarmController
from ui.dispatcher import ControlDispatcher
from ui.scene_spec import SceneValidationError, export_scene_text, parse_scene_text, validate_and_normalize
from ui.scenes import delete_scene, list_models, save_scene
from ui.telemetry_schema import write_schema


class WebSession:
    def __init__(
        self,
        *,
        screen_size: int = 900,
        max_steps: int = 600,
        goal_radius: float = 3.0,
        seed: int = 0,
        fps: int = 20,
        policy_workers: int = 0,
        oracle_enabled: bool = False,
        oracle_async: bool = True,
        oracle_update_interval: int = 10,
        attention_stride: int = 4,
        compare: bool = False,
        scene_root: Path | None = None,
        user_scene_root: Path | None = None,
        runs_root: Path | None = None,
    ):
        policy_workers = self._auto_policy_workers(policy_workers)
        self.ui_cfg = UIConfig(
            screen_size=screen_size,
            max_steps=max_steps,
            goal_radius=goal_radius,
            seed=seed,
            fps=fps,
            policy_workers=policy_workers,
            oracle_enabled=oracle_enabled,
            oracle_async=oracle_async,
            oracle_update_interval=oracle_update_interval,
            attention_stride=int(attention_stride),
        )
        self.scene_root = resolve_repo_path("scenarios") if scene_root is None else scene_root
        self.user_scene_root = resolve_repo_path("scenarios/user") if user_scene_root is None else user_scene_root
        self.seed = int(seed)
        self.controllers: dict[str, SwarmController] = {
            "left": SwarmController(self.ui_cfg, scene_root=self.scene_root, user_scene_root=self.user_scene_root),
        }
        self.compare = bool(compare)
        if self.compare:
            self._ensure_controller("right")
        self.paused = False
        self.step_once = False
        self.fps = int(max(1, fps))
        self.runs_root = resolve_repo_path("runs") if runs_root is None else Path(runs_root)
        self.models = list_models(self.runs_root)
        self.last_error = ""
        self._lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def _auto_policy_workers(self, value: int) -> int:
        if value > 0:
            return int(value)
        if value == 0:
            return 0
        cpu = os.cpu_count() or 2
        workers = int(min(12, max(1, cpu)))
        if workers <= 1:
            return 0
        return workers

    def _active_sides(self) -> list[str]:
        if self.compare:
            self._ensure_controller("right")
            return ["left", "right"]
        return ["left"]

    def _ensure_controller(self, side: str) -> SwarmController:
        side = "right" if side == "right" else "left"
        ctrl = self.controllers.get(side)
        if ctrl is not None:
            return ctrl
        ctrl = SwarmController(self.ui_cfg, scene_root=self.scene_root, user_scene_root=self.user_scene_root)
        left = self.controllers["left"]
        ctrl.selected_scene = left.selected_scene
        ctrl.custom_scene = dict(left.custom_scene) if isinstance(left.custom_scene, dict) else None
        ctrl.agent_idx = int(left.agent_idx)
        ctrl.overlay.show_grid = left.overlay.show_grid
        ctrl.overlay.show_trails = left.overlay.show_trails
        ctrl.overlay.show_threats = left.overlay.show_threats
        ctrl.overlay.show_attention = bool(getattr(left.overlay, "show_attention", False))
        ctrl.attention_channel = left.attention_channel
        ctrl.set_policy("baseline:astar_grid")
        ctrl.reset_env(new_map=False, seed=self.seed)
        self.controllers[side] = ctrl
        return ctrl

    def get_controller(self, side: str) -> SwarmController:
        return self._ensure_controller(side)

    def reset_env(self, new_map: bool = False) -> None:
        for side in self._active_sides():
            self.controllers[side].reset_env(new_map=new_map, seed=self.seed)

    def step_env(self) -> list[str]:
        done_sides: list[str] = []
        for side in self._active_sides():
            if self.controllers[side].step_env():
                done_sides.append(side)
        return done_sides

    def get_telemetry(self) -> dict[str, Any]:
        return {side: self.controllers[side].get_telemetry_dict() for side in self._active_sides()}

    def get_state(self) -> dict[str, Any]:
        return {
            "compare": self.compare,
            "paused": self.paused,
            "fps": self.fps,
            "seed": self.seed,
            "last_error": self.last_error,
            "left": asdict(self.controllers["left"].get_state()),
            "right": asdict(self.controllers["right"].get_state()) if "right" in self.controllers else None,
        }

    def reload_scenes(self) -> None:
        for ctrl in self.controllers.values():
            ctrl.reload_scenes()

    async def tick_and_render(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        async with self._lock:
            # Лок защищает от гонок между шагом симуляции и обновлением UI.
            done_sides: list[str] = []
            if (not self.paused) or self.step_once:
                loop = asyncio.get_running_loop()
                done_sides = await loop.run_in_executor(self._executor, self.step_env)
                self.step_once = False
            events: list[dict[str, Any]] = []
            if done_sides:
                for side in done_sides:
                    # Автосброс нужен, чтобы интерфейс не зависал в терминальном состоянии.
                    self.controllers[side].reset_env(new_map=False, seed=self.seed)
            return self.get_telemetry(), events

    def close(self) -> None:
        for ctrl in self.controllers.values():
            try:
                ctrl.close()
            except Exception:
                pass
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _scene_error(msg: str) -> dict[str, Any]:
    return {"type": "scene_error", "error": msg}


def _side_from_msg(msg: dict[str, Any]) -> str:
    return str(msg.get("side", "left"))


def _handle_pause(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    session.paused = not session.paused
    return []


def _handle_step(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    session.step_once = True
    return []


def _handle_reset(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    session.reset_env(new_map=False)
    return []


def _handle_new_map(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    session.seed += 1
    session.reset_env(new_map=True)
    return []


def _handle_compare(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    session.compare = bool(msg.get("value", session.compare))
    if session.compare:
        session._ensure_controller("right")
        session.reset_env(new_map=False)
    return []


def _handle_policy(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    name = msg.get("name", "")
    side = _side_from_msg(msg)
    if name:
        try:
            session.get_controller(side).set_policy(name)
            session.last_error = ""
        except Exception as exc:
            session.last_error = str(exc)
    return []


def _handle_scene(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    name = msg.get("name", "")
    for ctrl in session.controllers.values():
        ctrl.set_scene(name)
    return []


def _handle_seed(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        session.seed = int(msg.get("value", session.seed))
    except Exception:
        pass
    return []


def _handle_fps(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        session.fps = int(msg.get("value", session.fps))
    except Exception:
        pass
    return []


def _handle_agent(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        val = int(msg.get("value", 0))
    except Exception:
        val = 0
    for ctrl in session.controllers.values():
        ctrl.agent_idx = val
    return []


def _handle_model(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    val = msg.get("value") or ""
    det = bool(msg.get("deterministic", True))
    side = _side_from_msg(msg)
    if val:
        try:
            session.get_controller(side).set_model(str(val), deterministic=det)
            session.last_error = ""
        except Exception as exc:
            session.last_error = str(exc)
    return []


def _handle_deterministic(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    det = bool(msg.get("value", True))
    side = _side_from_msg(msg)
    session.get_controller(side).set_deterministic(det)
    return []


def _handle_toggle(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    name = msg.get("name")
    val = bool(msg.get("value"))
    for ctrl in session.controllers.values():
        if name == "show_grid":
            ctrl.overlay.show_grid = val
        elif name == "show_trails":
            ctrl.overlay.show_trails = val
        elif name == "show_threats":
            ctrl.overlay.show_threats = val
        elif name == "show_attention":
            ctrl.overlay.show_attention = val
    return []


def _handle_attention_channel(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    raw = str(msg.get("value", "sum")).lower()
    if raw not in {"sum", "x", "y"}:
        raw = "sum"
    for ctrl in session.controllers.values():
        ctrl.attention_channel = raw
        try:
            ctrl._attention_cache["step"] = -999
            ctrl._attention_cache["channel"] = None
        except Exception:
            pass
    return []


def _handle_oracle(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    enabled = msg.get("enabled")
    async_mode = msg.get("async")
    interval = msg.get("interval")
    for ctrl in session.controllers.values():
        ctrl.set_oracle(
            enabled=enabled if enabled is None else bool(enabled),
            async_mode=async_mode if async_mode is None else bool(async_mode),
            interval=interval,
        )
    return []


def _handle_tune(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    params = msg.get("params") or {}
    for ctrl in session.controllers.values():
        ctrl.apply_tunables(params)
    return []


def _handle_scene_preview(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    scene = msg.get("scene")
    res = validate_and_normalize(scene, allow_missing_id=True)
    if res.errors:
        err = "; ".join(res.errors)
        session.last_error = err
        responses.append(_scene_error(err))
    else:
        for ctrl in session.controllers.values():
            ctrl.set_custom_scene(res.scene)
            ctrl.reset_env(new_map=False, seed=session.seed)
    return responses


def _handle_scene_save(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    scene = msg.get("scene")
    res = validate_and_normalize(scene, allow_missing_id=True)
    if res.errors:
        err = "; ".join(res.errors)
        session.last_error = err
        responses.append(_scene_error(err))
    else:
        try:
            path = save_scene(res.scene or {}, session.user_scene_root)
        except SceneValidationError as exc:
            session.last_error = str(exc)
            responses.append(_scene_error(str(exc)))
        else:
            session.last_error = ""
            session.reload_scenes()
            responses.append({"type": "scenes", "scenes": session.controllers["left"].scene_names})
            responses.append({"type": "scene_saved", "scene_id": path.stem})
            for ctrl in session.controllers.values():
                ctrl.set_scene(path.stem)
    return responses


def _handle_scene_delete(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    scene_id = msg.get("scene_id") or msg.get("id") or ""
    if scene_id:
        removed = delete_scene(scene_id, session.user_scene_root)
        if not removed:
            responses.append(_scene_error("scene not found"))
    session.reload_scenes()
    responses.append({"type": "scenes", "scenes": session.controllers["left"].scene_names})
    return responses


def _handle_scene_refresh(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    session.reload_scenes()
    return [{"type": "scenes", "scenes": session.controllers["left"].scene_names}]


def _handle_scene_parse(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    text = msg.get("text", "")
    try:
        raw = parse_scene_text(text)
    except SceneValidationError as exc:
        return [_scene_error(str(exc))]
    res = validate_and_normalize(raw, allow_missing_id=True)
    if res.errors:
        return [_scene_error("; ".join(res.errors))]
    return [{"type": "scene_parsed", "scene": res.scene}]


def _handle_scene_export(session: WebSession, msg: dict[str, Any]) -> list[dict[str, Any]]:
    scene = msg.get("scene")
    fmt = msg.get("format", "json")
    res = validate_and_normalize(scene, allow_missing_id=True)
    if res.errors:
        return [_scene_error("; ".join(res.errors))]
    try:
        text = export_scene_text(res.scene or {}, fmt=fmt)
    except SceneValidationError as exc:
        return [_scene_error(str(exc))]
    return [{"type": "scene_exported", "format": fmt, "text": text}]


ACTION_HANDLERS = {
    "pause": _handle_pause,
    "step": _handle_step,
    "reset": _handle_reset,
    "new_map": _handle_new_map,
    "compare": _handle_compare,
    "policy": _handle_policy,
    "scene": _handle_scene,
    "seed": _handle_seed,
    "fps": _handle_fps,
    "agent": _handle_agent,
    "model": _handle_model,
    "deterministic": _handle_deterministic,
    "toggle": _handle_toggle,
    "oracle": _handle_oracle,
    "tune": _handle_tune,
    "scene_preview": _handle_scene_preview,
    "scene_save": _handle_scene_save,
    "scene_delete": _handle_scene_delete,
    "scene_refresh": _handle_scene_refresh,
    "scene_parse": _handle_scene_parse,
    "scene_export": _handle_scene_export,
    "attention_channel": _handle_attention_channel,
}

CONTROL_DISPATCHER = ControlDispatcher(ACTION_HANDLERS)


def handle_control_message(session: WebSession, msg: dict) -> list[dict[str, Any]]:
    return CONTROL_DISPATCHER.dispatch(session, msg)


def create_app(
    session_factory: WebSession | Callable[[], WebSession],
    static_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI()
    if isinstance(session_factory, WebSession):

        def _factory() -> WebSession:
            return session_factory
    else:
        _factory = session_factory

    if static_dir is None:
        static_dir = resolve_repo_path("ui_frontend/dist")
    try:
        write_schema(static_dir / "telemetry_schema.json")
    except Exception:
        pass
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    artifacts_dir = resolve_repo_path("runs")
    app.mount("/artifacts", StaticFiles(directory=str(artifacts_dir)), name="artifacts")
    index_path = static_dir / "index.html"
    favicon_path = static_dir / "favicon.ico"

    @app.get("/")
    def index():
        if not index_path.exists():
            return HTMLResponse("<h3>UI bundle not found</h3>", status_code=500)
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/favicon.ico")
    def favicon():
        if favicon_path.exists():
            return FileResponse(favicon_path)
        return HTMLResponse(status_code=404)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        session = _factory()
        use_binary = False
        await ws.send_json(
            {
                "type": "init",
                "policies": session.controllers["left"].available,
                "scenes": session.controllers["left"].scene_names,
                "models": session.models,
                "state": session.get_state(),
            }
        )

        async def sender():
            nonlocal use_binary
            while True:
                telemetry, events = await session.tick_and_render()
                for event in events:
                    await ws.send_json(event)
                payload = {
                    "type": "frame",
                    "telemetry": telemetry,
                    "n_agents_left": session.controllers["left"].env.n_agents,
                    "n_agents_right": session.controllers["right"].env.n_agents
                    if "right" in session.controllers
                    else 0,
                }
                if use_binary and msgpack is not None:
                    packed = msgpack.packb(payload, use_bin_type=True)
                    await ws.send_bytes(packed)
                else:
                    await ws.send_json(payload)
                await asyncio.sleep(1.0 / float(session.fps))

        async def receiver():
            nonlocal use_binary
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "hello":
                    use_binary = bool(msg.get("binary")) and (msgpack is not None)
                    continue
                if msg.get("type") != "control":
                    continue
                async with session._lock:
                    responses = handle_control_message(session, msg)
                for resp in responses:
                    await ws.send_json(resp)
                await ws.send_json({"type": "state", "state": session.get_state()})

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        try:
            _done, pending = await asyncio.wait(
                {send_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
        except WebSocketDisconnect:
            send_task.cancel()
            recv_task.cancel()
        finally:
            session.close()

    return app


@hydra.main(version_base=None, config_path="../configs/hydra/ui", config_name="web")
def main(cfg) -> None:
    cfg = apply_schema(cfg, WebUIConfig)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, dict):
        raise SystemExit("Некорректный формат конфигурации Web UI.")

    scene_root = resolve_repo_path(cfg["scene_root"]) if cfg.get("scene_root") else None
    user_scene_root = resolve_repo_path(cfg["user_scene_root"]) if cfg.get("user_scene_root") else None
    runs_root = resolve_repo_path(cfg["runs_root"]) if cfg.get("runs_root") else None

    def _session_factory() -> WebSession:
        return WebSession(
            screen_size=int(cfg["screen_size"]),
            max_steps=int(cfg["max_steps"]),
            goal_radius=float(cfg["goal_radius"]),
            seed=int(cfg["seed"]),
            fps=int(cfg["fps"]),
            policy_workers=int(cfg["policy_workers"]),
            oracle_enabled=bool(cfg.get("oracle_enabled", False)),
            oracle_async=bool(cfg.get("oracle_async", True)),
            oracle_update_interval=int(cfg.get("oracle_update_interval", 10)),
            attention_stride=int(cfg.get("attention_stride", 4)),
            compare=bool(cfg["compare"]),
            scene_root=scene_root,
            user_scene_root=user_scene_root,
            runs_root=runs_root,
        )

    static_dir = resolve_repo_path("ui_frontend/dist")
    if not static_dir.exists():
        raise SystemExit("React UI bundle not found. Run `make ui-build`.")
    app = create_app(_session_factory, static_dir=static_dir)

    try:
        import uvicorn
    except Exception as exc:
        raise SystemExit(f"uvicorn is required to run web UI: {exc}") from exc

    uvicorn.run(app, host=str(cfg["host"]), port=int(cfg["port"]), log_level="info")


if __name__ == "__main__":
    main()
