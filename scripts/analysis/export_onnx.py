"""Экспорт PPO-политики в ONNX и INT8 (ONNX Runtime quantization)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import onnx
except Exception:
    onnx = None

try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except Exception:
    quantize_dynamic = None
    QuantType = None

try:
    from stable_baselines3 import PPO
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"stable_baselines3 is required: {exc}") from exc


@dataclass
class ExportOnnxConfig:
    model_path: str = ""
    out_dir: str = "runs/exports"
    out_name: str = ""
    opset: int = 17
    quantize_int8: bool = True
    dynamic_axes: bool = True
    device: str = "cpu"


class SB3OnnxWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.set_training_mode(False)

    def forward(self, vector: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        obs = {"vector": vector, "grid": grid}
        if getattr(self.policy, "share_features_extractor", True):
            features = self.policy.extract_features(obs)
        else:
            features = self.policy.pi_features_extractor(obs)
        latent_pi, _ = self.policy.mlp_extractor(features)
        return self.policy.action_net(latent_pi)


def _resolve_out_name(model_path: Path, out_name: str | None) -> str:
    if out_name:
        return out_name
    stem = model_path.stem or "policy"
    return f"{stem}_ppo"


def _export_onnx(cfg: ExportOnnxConfig) -> None:
    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise SystemExit(f"model_path not found: {model_path}")
    if onnx is None:
        raise SystemExit("onnx is not installed (pip install onnx onnxruntime)")

    model = PPO.load(str(model_path), device=cfg.device)
    policy = model.policy
    policy.set_training_mode(False)

    obs_space = getattr(model, "observation_space", None)
    if obs_space is None or not hasattr(obs_space, "spaces"):
        raise SystemExit("Only Dict observation space is supported for export (vector/grid).")
    if "vector" not in obs_space.spaces or "grid" not in obs_space.spaces:
        raise SystemExit("Observation space must contain keys: vector, grid.")

    vec_space = obs_space.spaces["vector"]
    grid_space = obs_space.spaces["grid"]
    vec_dim = int(vec_space.shape[0])
    grid_shape = tuple(int(x) for x in grid_space.shape)

    wrapper = SB3OnnxWrapper(policy)
    wrapper.eval()

    device = torch.device(cfg.device)
    vector = torch.zeros((1, vec_dim), dtype=torch.float32, device=device)
    grid = torch.zeros((1, *grid_shape), dtype=torch.float32, device=device)

    out_root = Path(cfg.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    base = _resolve_out_name(model_path, cfg.out_name)
    onnx_path = out_root / f"{base}.onnx"

    dynamic_axes = None
    if cfg.dynamic_axes:
        dynamic_axes = {
            "vector": {0: "batch"},
            "grid": {0: "batch"},
            "action": {0: "batch"},
        }

    torch.onnx.export(
        wrapper,
        (vector, grid),
        str(onnx_path),
        input_names=["vector", "grid"],
        output_names=["action"],
        dynamic_axes=dynamic_axes,
        opset_version=int(cfg.opset),
    )

    meta = {
        "model_path": str(model_path),
        "onnx_path": str(onnx_path),
        "vector_dim": vec_dim,
        "grid_shape": list(grid_shape),
        "opset": int(cfg.opset),
        "quantize_int8": bool(cfg.quantize_int8),
    }

    if cfg.quantize_int8:
        if quantize_dynamic is None or QuantType is None:
            raise SystemExit("onnxruntime is required for quantization (pip install onnxruntime)")
        q_path = out_root / f"{base}_int8.onnx"
        quantize_dynamic(
            model_input=str(onnx_path),
            model_output=str(q_path),
            weight_type=QuantType.QInt8,
        )
        meta["onnx_int8_path"] = str(q_path)

    meta_path = out_root / f"{base}_export.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] {onnx_path}")
    if meta.get("onnx_int8_path"):
        print(f"[OK] {meta['onnx_int8_path']}")
    print(f"[OK] {meta_path}")


def main() -> None:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    from scripts.common.hydra_utils import apply_schema

    @hydra.main(version_base=None, config_path="../../configs/hydra", config_name="analysis/export_onnx")
    def _run(cfg: DictConfig) -> None:
        cfg = apply_schema(cfg, ExportOnnxConfig)
        data = OmegaConf.to_container(cfg, resolve=True)
        _export_onnx(ExportOnnxConfig(**data))

    _run()


if __name__ == "__main__":
    main()
