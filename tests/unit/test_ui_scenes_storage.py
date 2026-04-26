from pathlib import Path

from ui.scenes import delete_scene, load_scenes, save_scene


def test_save_load_delete_scene(tmp_path: Path):
    scene = {
        "id": "custom_test_scene",
        "field_size": 100,
        "start_center": [10, 10],
        "target_pos": [90, 90],
        "walls": [[20, 20, 30, 30]],
        "threats": [{"pos": [50, 50], "radius": 5, "intensity": 0.1}],
        "max_steps": 300,
    }
    path = save_scene(scene, tmp_path)
    assert path.exists()

    scenes = load_scenes(tmp_path)
    ids = [name for name, _ in scenes]
    assert "custom_test_scene" in ids

    removed = delete_scene("custom_test_scene", tmp_path)
    assert removed
    assert not path.exists()
