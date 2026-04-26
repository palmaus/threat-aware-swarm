from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf

from env.config import EnvConfig as BaseEnvConfig


@dataclass
class EnvConfig(BaseEnvConfig):
    """Hydra train schema extension; split runtime-only fields before EnvConfig.from_dict()."""

    max_steps: int = 600
    goal_radius: float = 3.0


@dataclass
class RunConfig:
    run_name: str = ""
    seed: int = 0
    total_timesteps: int = 8_000_000
    eval_freq: int = 200_000
    n_eval_episodes: int = 20
    checkpoint_freq: int = 1_000_000
    out_dir: str = "runs"
    deterministic_eval: bool = False
    no_eval: bool = False


@dataclass
class VecConfig:
    num_vec_envs: int = 1
    num_cpus: int = 1


@dataclass
class PpoConfig:
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 2048
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    clip_range: float = 0.2
    target_kl: float = 0.01
    net_arch: list[int] = field(default_factory=lambda: [256, 256])
    log_std_init: float = -1.0
    cnn_features_dim: int = 256
    cnn_extractor: str = "advanced"
    use_rnn: bool = True
    lstm_hidden_size: int = 256
    enable_critic_lstm: bool = True


@dataclass
class ADRConfig:
    enabled: bool = False
    metric: str = "swarm/finished_frac"
    threshold: float = 0.8
    interval_steps: int = 50_000
    step: float = 0.1
    params: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class ALPConfig:
    enabled: bool = False
    metric: str = "swarm/finished_frac"
    interval_steps: int = 50_000
    epsilon: float = 0.1
    buckets: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CurriculumConfig:
    enabled: bool = False
    config: str = "configs/curriculum/base.yaml"
    profile: str = "default"
    metric: str = "swarm/finished_frac"
    threshold: float = 0.7
    min_steps: int = 200_000
    ema_alpha: float = 0.2
    goal_radius_decay_steps: int = 500_000
    adr: ADRConfig = field(default_factory=ADRConfig)
    alp: ALPConfig = field(default_factory=ALPConfig)


@dataclass
class ResumeConfig:
    enabled: bool = False
    run_dir: str = ""
    run_name: str = ""
    model: str = ""
    apply_overrides: bool = True
    warn_on_mismatch: bool = True


@dataclass
class MlflowConfig:
    enabled: bool = False
    tracking_uri: str = "http://localhost:5000"
    experiment: str = "swarm"
    register_model_name: str = ""
    log_tensorboard: bool = True
    cleanup_run_dir: bool = False
    tags: list[str] = field(default_factory=list)


@dataclass
class ClearMLConfig:
    enabled: bool = False
    project: str = "Threat-Aware-Swarm"
    task_name: str = "train_ppo"
    tags: list[str] = field(default_factory=list)
    output_uri: str = ""


@dataclass
class TrackingConfig:
    log_every_steps: int = 2048
    system_metrics: dict = field(default_factory=lambda: {"enabled": False})
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)
    clearml: ClearMLConfig = field(default_factory=ClearMLConfig)


@dataclass
class LoggingConfig:
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    hydra: dict = field(default_factory=dict)


@dataclass
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    vec: VecConfig = field(default_factory=VecConfig)
    ppo: PpoConfig = field(default_factory=PpoConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    experiment: dict = field(default_factory=dict)
    device: str = "auto"
    exp_id: str = ""
    progress_bar: bool = False


def apply_schema(cfg):
    """Проверяет Hydra‑конфиг с помощью структурированной схемы."""
    schema = OmegaConf.structured(TrainConfig())
    merged = OmegaConf.merge(schema, cfg)
    OmegaConf.set_struct(merged, True)
    return merged
