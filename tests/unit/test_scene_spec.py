from ui.scene_spec import export_scene_text, parse_scene_text, validate_and_normalize


def test_validate_and_normalize_ok():
    scene = {
        "id": "T_scene",
        "field_size": 50,
        "start_center": [5, 5],
        "target_pos": [40, 40],
        "walls": [[10, 10, 12, 20]],
        "threats": [{"pos": [25, 25], "radius": 3, "intensity": 0.2}],
        "max_steps": 123,
    }
    res = validate_and_normalize(scene)
    assert res.errors == []
    assert res.scene["id"] == "T_scene"
    assert res.scene["field_size"] == 50.0
    assert res.scene["max_steps"] == 123
    assert res.scene["walls"][0] == [10.0, 10.0, 12.0, 20.0]
    assert res.scene["threats"][0]["radius"] == 3.0


def test_validate_and_normalize_errors():
    scene = {
        "field_size": 10,
        "start_center": [0, 0],
        "target_pos": [20, 20],
        "threats": [{"pos": [5, 5], "radius": -1, "intensity": 0.1}],
    }
    res = validate_and_normalize(scene)
    assert res.scene is None
    assert any("id is required" in e for e in res.errors)
    assert any("target_pos" in e or "out of bounds" in e for e in res.errors)


def test_parse_and_export_roundtrip():
    raw = """{\n  \"id\": \"T\",\n  \"field_size\": 100,\n  \"start_center\": [10, 10],\n  \"target_pos\": [90, 90]\n}\n"""
    data = parse_scene_text(raw)
    res = validate_and_normalize(data)
    assert res.errors == []
    text = export_scene_text(res.scene, fmt="json")
    data2 = parse_scene_text(text)
    res2 = validate_and_normalize(data2)
    assert res2.errors == []


def test_dynamic_threat_fields_preserved():
    scene = {
        "id": "dyn",
        "field_size": 100,
        "start_center": [10, 10],
        "target_pos": [90, 90],
        "threats": [
            {"pos": [50, 50], "radius": 5, "intensity": 0.2, "type": "linear", "speed": 2.5, "angle": 45},
            {"pos": [60, 60], "radius": 4, "intensity": 0.1, "type": "brownian", "speed": 1.2, "noise_scale": 0.3},
            {"pos": [70, 70], "radius": 6, "intensity": 0.15, "type": "chaser", "speed": 1.0, "vision_radius": 25},
        ],
    }
    res = validate_and_normalize(scene)
    assert res.errors == []
    t0 = res.scene["threats"][0]
    t1 = res.scene["threats"][1]
    t2 = res.scene["threats"][2]
    assert t0["speed"] == 2.5
    assert t0["angle"] == 45.0
    assert t1["noise_scale"] == 0.3
    assert t2["vision_radius"] == 25.0


def test_dynamic_threats_are_validated_when_saved_from_ui():
    scene = {
        "id": "bad_dyn",
        "field_size": 100,
        "start_center": [10, 10],
        "target_pos": [90, 90],
        "dynamic_threats": [
            {"type": "linear", "pos": [150, 150], "radius": -5, "intensity": 0.1},
        ],
    }

    res = validate_and_normalize(scene)

    assert res.scene is None
    assert any("dynamic_threats" in err for err in res.errors)


def test_start_centers_and_agents_pos_are_bounds_checked():
    bad_centers = {
        "id": "bad_centers",
        "field_size": 100,
        "start_centers": [[-10, 10], [20, 20]],
        "target_pos": [90, 90],
    }
    bad_agents = {
        "id": "bad_agents",
        "field_size": 100,
        "agents_pos": [[999, 999]],
        "target_pos": [90, 90],
    }

    centers_res = validate_and_normalize(bad_centers)
    agents_res = validate_and_normalize(bad_agents)

    assert centers_res.scene is None
    assert any("start_centers[0]" in err for err in centers_res.errors)
    assert agents_res.scene is None
    assert any("agents_pos[0]" in err for err in agents_res.errors)
