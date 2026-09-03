from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.classifier import MSELossConfig
from prox_lora.models.ConvNet import KaggleConvNetConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

v2_baseline = FullTrainConfig(
    name="conv2_baseline",
    wandb_project="conv_dr",
    datamodule=DRConfig(augmentations=True, size=512, weighted_sampler="sqrt"),
    dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
    model=KaggleConvNetConfig(input_shape=(3, 512, 512), channels=(32, 64, 128, 256), num_classes=5),
    optimizer=OptimizerConfig(opt="adamw", lr=1e-3, weight_decay=1e-4),  # 1e-3 .. 2e-3 works best.
    scheduler=SchedulerConfig(sched="cosine", num_epochs=75, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
    trainer=TrainerConfig(precision="bf16-mixed", max_epochs=75, log_every_n_steps=100),
    loss_ce_alpha=0.2,
    loss_class_weights_gamma=0.0,
    mse_loss=MSELossConfig(alpha=5, use_class_weights=True),
)
v2_b32 = deep_replace(v2_baseline, {"name": "conv2_b32", "dataloader.batch_size": 32})

v2_sgd = deep_replace(
    v2_baseline,
    # lr=7e-3 .. 2e-1 works best.
    {"name": "conv2_sgd", "optimizer": OptimizerConfig(opt="sgd", lr=5e-2, weight_decay=1e-4, momentum=0.9)},
)
v2_sgd_b32 = deep_replace(v2_sgd, {"name": "conv2_sgd_b32", "dataloader.batch_size": 32})

v2_proxsamadw = deep_replace(
    v2_baseline,
    {
        "name": "conv2_proxsamadw",
        "optimizer": OptimizerConfig(opt="proxsamadw", lr=1e-3, weight_decay=1e-4, momentum=0.9, prox_lambda=0, rho=0),
    },
)

register_configs(
    v2_baseline,
    v2_b32,
    v2_sgd,
    v2_sgd_b32,
    # --------------------------------------------------#
    # AdamW
    v2_proxsamadw,
    # --------------------------------------------------#
    # Basic prox - ISTA, FISTA
    deep_replace(
        v2_sgd, {"name": "conv2_ista", "optimizer.opt": "ista", "optimizer.lr": 5e-2, "optimizer.prox_lambda": 1e-4}
    ),
)
