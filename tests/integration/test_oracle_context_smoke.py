from __future__ import annotations

import numpy as np

from common.context import context_from_state
from env.config import EnvConfig
from env.engine import SwarmEngine


def test_oracle_context_smoke():
    cfg = EnvConfig()
    cfg.oracle_enabled = True
    engine = SwarmEngine(cfg, max_steps=5, goal_radius=3.0)
    engine.reset(seed=0)
    public_state = engine.get_public_state(include_oracle=True)
    ctx = context_from_state(public_state, include_oracle=True)
    assert np.asarray(ctx.pos).shape[1] == 2
