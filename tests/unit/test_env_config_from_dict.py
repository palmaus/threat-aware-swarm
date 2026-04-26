from env.config import EnvConfig


def test_from_dict_keeps_valid_nested_values_when_unknown_keys_exist() -> None:
    cfg = EnvConfig.from_dict(
        {
            "physics": {"mass": 2.0, "unknown": "ignored"},
            "wind": {"enabled": False, "unknown": "ignored"},
            "battery": {"capacity": 42.0, "unknown": "ignored"},
            "not_a_runtime_key": "ignored",
        }
    )

    assert cfg.physics.mass == 2.0
    assert cfg.wind.enabled is False
    assert cfg.battery.capacity == 42.0
