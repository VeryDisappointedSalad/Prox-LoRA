from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.classifier import MSELossConfig
from prox_lora.models.ConvNet import KaggleConvNetConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

conv4_adamw = FullTrainConfig(
    name="conv4_adamw",
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

# As the SGD baseline, use non-Nesterov momentum with decoupled weight decay (SGDW).
conv4_sgd = deep_replace(
    conv4_adamw,
    # lr=2e-2 .. 2e-1, wd<=5e-3, pl<=2e-4, rho=2e-2 (or <=1e-1) works best.
    {
        "name": "conv4_sgd",
        "optimizer": OptimizerConfig(opt="sgdw-nonesterov", lr=5e-2, weight_decay=5e-3, momentum=0.9),
    },
)

conv4_proxsam_dummy = deep_replace(
    conv4_sgd,
    {
        "name": "conv4_proxsam_dummy",
        "optimizer": OptimizerConfig(opt="proxsam", prox_lambda=0, rho=0),
    },
)

register_configs(
    conv4_adamw,
    conv4_sgd,
    conv4_proxsam_dummy,
    # deep_replace(conv4_sgd, {"name": "conv4_sgd_m0", "optimizer.momentum": 0.0}),
)
