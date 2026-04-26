import json
from pathlib import Path

from ui.web_server import WebSession, handle_control_message


def _base_scene():
    return {
        "id": "S0",
        "field_size": 100,
        "start_center": [10, 10],
        "target_pos": [90, 90],
        "threats": [],
        "walls": [],
        "max_steps": 200,
    }


def test_scene_save_and_refresh(tmp_path: Path):
    scene_root = tmp_path / "scenarios"
    user_root = tmp_path / "user"
    scene_root.mkdir(parents=True)
    (scene_root / "S0.json").write_text(json.dumps(_base_scene()), encoding="utf-8")

    session = WebSession(scene_root=scene_root, user_scene_root=user_root)
    new_scene = {
        "id": "custom_ui",
        "field_size": 100,
        "start_center": [10, 10],
        "target_pos": [90, 90],
        "threats": [],
        "walls": [],
        "max_steps": 300,
    }
    responses = handle_control_message(session, {"type": "control", "action": "scene_save", "scene": new_scene})
    assert (user_root / "custom_ui.json").exists()
    assert any(r.get("type") == "scenes" and "custom_ui" in r.get("scenes", []) for r in responses)


def test_scene_parse_and_export(tmp_path: Path):
    scene_root = tmp_path / "scenarios"
    user_root = tmp_path / "user"
    scene_root.mkdir(parents=True)
    (scene_root / "S0.json").write_text(json.dumps(_base_scene()), encoding="utf-8")

    session = WebSession(scene_root=scene_root, user_scene_root=user_root)
    raw = json.dumps(_base_scene())
    responses = handle_control_message(session, {"type": "control", "action": "scene_parse", "text": raw})
    parsed = next(r for r in responses if r.get("type") == "scene_parsed")
    assert parsed["scene"]["id"] == "S0"

    responses = handle_control_message(
        session,
        {"type": "control", "action": "scene_export", "scene": parsed["scene"], "format": "json"},
    )
    exported = next(r for r in responses if r.get("type") == "scene_exported")
    assert "\"id\": \"S0\"" in exported["text"]
