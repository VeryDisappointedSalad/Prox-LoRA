from collections.abc import Collection, Sequence
from dataclasses import dataclass
from typing import Literal, Required, TypedDict

from prox_lora.infrastructure.configs import yaml


class OptimizerConfig(TypedDict, total=False):
    """
    Optimizer kwargs, as used by timm.optim.create_optimizer_v2(model, **kwargs).

    Extra optimizer-specific arguments (like betas for Adam) can be provided too.

    Note that parameter groups can also be specified with create_optimizer_v2().

    See: https://huggingface.co/docs/timm/reference/optimizers#timm.optim.create_optimizer_v2
    """

    opt: Required[str]
    """
    Optimizer type, e.g., "sgd", "adamw", "lion", and many variants.

    See:
    - https://github.com/huggingface/pytorch-image-models/tree/main/timm/optim
    - timm.optim.list_optimizers()
    - https://timm.fast.ai/Optimizers (incomplete).

    Note that "sgd" means SGD with Nesterov momentum in timm (use opt="momentum" to disable Nesterov).

    New optimizers can be added with `timm.optim.default_registry.register()`.
    """

    lr: float | None
    """Learning rate. If None (default), will use the optimizer's default."""

    weight_decay: float
    """Weight decay factor. Default is 0 (no weight decay)."""

    filter_bias_and_bn: bool
    """
    If True (default), bias and norm layer parameters (all 1d params) will not have weight decay applied.

    Only used when no param groups are provided and weight_decay > 0.
    """

    momentum: float
    """Momentum factor, only used for optimizers that support it. Default is 0.9."""

    layer_decay: float | None
    """
    Optional layer-wise learning rate decay factor.
    If provided, learning rates will be scaled by layer_decay^(max_depth - layer_depth).
    Only used when model_or_params is a model.
    """

    layer_decay_min_scale: float  # Default: 0
    layer_decay_no_opt_scale: float

    fallback_list: Collection[str]
    """Parameter patterns to use fallback in hybrid optimizers (e.g., AdamW for Muon)."""

    fallback_no_weight_decay: bool
    """If True, params in no_weight_decay() list will use fallback optimizer; default is False."""


@yaml.register_class
@dataclass(frozen=True)
class SchedulerConfig:
    """
    Scheduler config, as used by timm.scheduler.create_scheduler_v2(opt, **as_dict(config)).

    See: https://huggingface.co/docs/timm/reference/schedulers#timm.scheduler.create_scheduler_v2
    """

    sched: Literal["none", "cosine", "tanh", "step", "multistep", "plateau", "poly"] = "cosine"

    num_epochs: int = 300
    decay_epochs: int = 90
    decay_milestones: Sequence[int] = (90, 180, 270)
    cooldown_epochs: int = 0
    patience_epochs: int = 10
    decay_rate: float = 0.1
    min_lr: float = 0
    warmup_lr: float = 1e-5
    warmup_epochs: int = 0
    warmup_prefix: bool = False

    noise: float | list[float] | None = None
    noise_pct: float = 0.67
    noise_std: float = 1.0
    noise_seed: int = 42
    cycle_mul: float = 1.0
    cycle_decay: float = 0.1
    cycle_limit: int = 1
    k_decay: float = 1.0
    plateau_mode: str = "max"
    step_on_epochs: bool = True
    updates_per_epoch: int = 0
