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
    compiler_backend="inductor",
)

conv4_proxsamadamw_dummy = deep_replace(
    conv4_adamw, {"name": "conv4_proxsamadamw_dummy", "optimizer": OptimizerConfig(opt="proxsamadw", prox_lambda=0, rho=0)}
)

conv4_proxsamadaptive_dummy = deep_replace(
    conv4_adamw,
    {"name": "conv4_proxsamadaptive_dummy", "optimizer": OptimizerConfig(opt="proxsamadaptive", prox_lambda=0, rho=0)},
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
    conv4_sgd,
    conv4_proxsam_dummy,
    # deep_replace(conv4_sgd, {"name": "conv4_sgd_m0", "optimizer.momentum": 0.0}),
    deep_replace(conv4_proxsam_dummy, {"name": "conv4_pl1e-4_rho2e-2", "optimizer.prox_lambda": 1e-4, "optimizer.rho": 2e-2}),

    # Running:
    deep_replace(conv4_proxsam_dummy, {"name": "conv4_pl1e-4_rho0", "optimizer.prox_lambda": 1e-4, "optimizer.rho": 0}),   # The one running is actually pl1e-4_rho2e-2 :/
    deep_replace(conv4_proxsam_dummy, {"name": "conv4_pl0_rho2e-2", "optimizer.prox_lambda": 0.0, "optimizer.rho": 2e-2}),

    # TODO:
    deep_replace(conv4_proxsam_dummy, {"name": "conv4_pl1e-4_rho2e-2_lrx2", "optimizer.prox_lambda": 1e-4, "optimizer.rho": 2e-2, "optimizer.lr": 1e-1}),
    deep_replace(conv4_proxsam_dummy, {"name": "conv4_pl1e-4_rho2e-2_wdv2", "optimizer.prox_lambda": 1e-4, "optimizer.rho": 2e-2, "optimizer.weight_decay": 2.5e-3}),
    conv4_adamw,
    conv4_proxsamadamw_dummy,
    conv4_proxsamadaptive_dummy,
    deep_replace(conv4_proxsamadamw_dummy, {"name": "conv4_proxsamadamw_pl0_rho2e-3", "optimizer.prox_lambda": 0, "optimizer.rho": 2e-3}),
    deep_replace(conv4_proxsamadamw_dummy, {"name": "conv4_proxsamadamw_pl1e-2_rho0", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 0}),
    deep_replace(conv4_proxsamadamw_dummy, {"name": "conv4_proxsamadamw_pl1e-2_rho2e-3", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 2e-3}),
    deep_replace(conv4_proxsamadaptive_dummy, {"name": "conv4_proxsamadaptive_pl0_rho2e-3", "optimizer.prox_lambda": 0, "optimizer.rho": 2e-3}),
    deep_replace(conv4_proxsamadaptive_dummy, {"name": "conv4_proxsamadaptive_pl1e-2_rho0", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 0}),
    deep_replace(conv4_proxsamadaptive_dummy, {"name": "conv4_proxsamadaptive_pl1e-2_rho2e-3", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 2e-3}),
)
