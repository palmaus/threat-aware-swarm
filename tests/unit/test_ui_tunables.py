from ui.config import UIConfig
from ui.controller import SwarmController


def test_ui_tunables_domain_randomization_and_noise():
    cfg = UIConfig(screen_size=200, max_steps=5, seed=0, fps=1)
    controller = SwarmController(cfg)

    controller.apply_tunables(
        {
            "max_accel": 12.5,
            "drag": 0.12,
            "obs_noise_target": 0.01,
            "obs_noise_vel": 0.02,
            "obs_noise_grid": 0.03,
            "domain_randomization": True,
            "dr_max_speed_min": 1.5,
            "dr_max_speed_max": 2.5,
            "dr_drag_min": 0.05,
            "dr_drag_max": 0.15,
        }
    )

    env_cfg = controller.env.config
    assert env_cfg.physics.max_thrust == 12.5
    assert env_cfg.physics.drag_coeff == 1.2
    assert env_cfg.obs_noise_target == 0.01
    assert env_cfg.obs_noise_vel == 0.02
    assert env_cfg.obs_noise_grid == 0.03
    assert env_cfg.domain_randomization is True
    assert env_cfg.dr_max_speed_min == 1.5
    assert env_cfg.dr_max_speed_max == 2.5
    assert env_cfg.dr_drag_min == 0.05
    assert env_cfg.dr_drag_max == 0.15

    tunables = controller.get_tunables()
    assert tunables["obs_noise_grid"] == 0.03
    assert tunables["domain_randomization"] is True
    assert tunables["dr_max_speed_min"] == 1.5
    assert tunables["dr_max_speed_max"] == 2.5
    assert tunables["dr_drag_min"] == 0.05
    assert tunables["dr_drag_max"] == 0.15
