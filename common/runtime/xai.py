"""Runtime saliency helpers for SB3 Dict-observation policies."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from captum.attr import Saliency
from stable_baselines3.common.policies import BasePolicy, MultiInputActorCriticPolicy


class SB3PolicyWrapper(torch.nn.Module):
    """Captum-compatible wrapper around an SB3 actor policy."""

    def __init__(self, policy: MultiInputActorCriticPolicy):
        super().__init__()
        self.policy = policy
        self.policy.set_training_mode(False)

    def forward(self, grid_tensor: torch.Tensor, vector_tensor: torch.Tensor) -> torch.Tensor:
        obs = {"grid": grid_tensor, "vector": vector_tensor}
        if getattr(self.policy, "share_features_extractor", True):
            features = self.policy.extract_features(obs)
        else:
            features = self.policy.pi_features_extractor(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi)


def _to_batch_grid(grid: torch.Tensor) -> torch.Tensor:
    if grid.ndim == 2:
        return grid.unsqueeze(0).unsqueeze(0)
    if grid.ndim == 3:
        return grid.unsqueeze(0)
    if grid.ndim == 4:
        return grid
    raise ValueError(f"Unexpected grid shape: {tuple(grid.shape)}")


def _to_batch_vector(vector: torch.Tensor) -> torch.Tensor:
    if vector.ndim == 1:
        return vector.unsqueeze(0)
    if vector.ndim == 2:
        return vector
    raise ValueError(f"Unexpected vector shape: {tuple(vector.shape)}")


def get_saliency_map(
    model_or_policy: Any,
    obs_dict: dict[str, Any],
    *,
    action_dim: int = 0,
    normalize: bool = True,
) -> np.ndarray:
    """Return a 2D heat map for the selected continuous action dimension."""

    if "grid" not in obs_dict or "vector" not in obs_dict:
        raise ValueError("obs_dict must contain 'grid' and 'vector'.")

    policy: BasePolicy = model_or_policy.policy if hasattr(model_or_policy, "policy") else model_or_policy
    if not hasattr(policy, "action_net"):
        raise ValueError("Policy does not expose action_net.")

    device = getattr(policy, "device", None)
    if device is None:
        device = next(policy.parameters()).device

    grid_tensor = torch.as_tensor(obs_dict["grid"], dtype=torch.float32, device=device)
    vector_tensor = torch.as_tensor(obs_dict["vector"], dtype=torch.float32, device=device)
    grid_tensor = _to_batch_grid(grid_tensor)
    vector_tensor = _to_batch_vector(vector_tensor)
    grid_tensor.requires_grad_(True)

    wrapper = SB3PolicyWrapper(policy)
    saliency = Saliency(wrapper)
    policy.zero_grad(set_to_none=True)

    grads = saliency.attribute(grid_tensor, additional_forward_args=(vector_tensor,), target=action_dim)
    grads_abs = grads.abs()
    if grads_abs.ndim == 4:
        grads_abs = grads_abs.squeeze(0)
    if grads_abs.ndim == 3:
        heatmap = grads_abs.sum(dim=0)
    elif grads_abs.ndim == 2:
        heatmap = grads_abs
    else:
        raise ValueError(f"Unexpected gradient shape: {tuple(grads_abs.shape)}")

    heatmap = heatmap.detach().cpu().numpy()
    if normalize:
        max_val = float(np.max(heatmap))
        if max_val > 1e-8:
            heatmap = heatmap / max_val
    return heatmap
