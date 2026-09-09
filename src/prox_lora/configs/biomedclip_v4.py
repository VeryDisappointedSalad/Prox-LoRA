from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.biomedclip import BiomedCLIPConfig
from prox_lora.models.classifier import MSELossConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

bmc4_adamw = FullTrainConfig(
    name="bmc4_adamw",
    wandb_project="biomedclip_dr",
    datamodule=DRConfig(augmentations=True, size=512, weighted_sampler="sqrt"),
    dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
    # unfrozen_groups=14 cały model, unfrozen_groups=0 tylko głowa
    model=BiomedCLIPConfig(input_shape=(3, 512, 512), num_classes=5, unfrozen_groups=14, freeze_bn_stats=False),
    optimizer=OptimizerConfig(opt="adamw", lr=5e-6, weight_decay=1e-4),
    scheduler=SchedulerConfig(
        sched="cosine", num_epochs=30, warmup_epochs=1, warmup_lr=1e-7, min_lr=1e-7, step_on_epochs=False
    ),
    trainer=TrainerConfig(
        precision="bf16-mixed",
        accumulate_grad_batches=1,  # >1 is incompatible with mixed precision.
        max_epochs=30,
        log_every_n_steps=100,
    ),
    loss_ce_alpha=0.2,
    loss_class_weights_gamma=0.0,
    mse_loss=MSELossConfig(alpha=5, use_class_weights=True),
    compiler_backend="inductor",
)

bmc4_proxsamadamw_dummy = deep_replace(
    bmc4_adamw, {"name": "bmc4_proxsamadamw_dummy", "optimizer": OptimizerConfig(opt="proxsamadw", prox_lambda=0, rho=0)}
)

bmc4_proxsamadaptive_dummy = deep_replace(
    bmc4_adamw,
    {"name": "bmc4_proxsamadaptive_dummy", "optimizer": OptimizerConfig(opt="proxsamadaptive", prox_lambda=0, rho=0)},
)

# As the SGD baseline, use non-Nesterov momentum with decoupled weight decay (SGDW).
# For SGD, both batch-size 32 and 64 work about the same, best for lr<=2e-5, wd<=1.0, pl<=2e-2, rho <5e-3
bmc4_sgd = deep_replace(
    bmc4_adamw,
    {"name": "bmc4_sgd", "optimizer": OptimizerConfig(opt="sgdw-nonesterov", lr=2e-5, weight_decay=1e-1, momentum=0.9)},
)


bmc4_proxsam_dummy = deep_replace(
    bmc4_sgd, {"name": "bmc4_proxsam_dummy", "optimizer": OptimizerConfig(opt="proxsam", prox_lambda=0, rho=0)}
)

register_configs(
    bmc4_adamw,
    # deep_replace(bmc4_baseline, {"name": "bmc4_1024", "datamodule.size": 1024, "model.input_shape": (3, 1024, 1024)}),
    # deep_replace(
    #     bmc4_baseline, {"name": "bmc4_nosampler", "datamodule.weighted_sampler": False, "loss_class_weights_gamma": -0.5}
    # ),
    bmc4_sgd,
    bmc4_proxsam_dummy,
    deep_replace(
        bmc4_proxsam_dummy, {"name": "bmc4_pl1e-2_rho2e-3", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 2e-3}
    ),
    deep_replace(bmc4_proxsam_dummy, {"name": "bmc4_pl1e-2_rho0", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 0}),
    deep_replace(bmc4_proxsam_dummy, {"name": "bmc4_pl0_rho2e-3", "optimizer.prox_lambda": 0, "optimizer.rho": 2e-3}),
    # Running:

    # Todo:
    bmc4_proxsamadamw_dummy,
    bmc4_proxsamadaptive_dummy,
    deep_replace(bmc4_proxsamadamw_dummy, {"name": "bmc4_proxsamadamw_pl1e-2_rho2e-3", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 2e-3}),
    deep_replace(bmc4_proxsamadamw_dummy, {"name": "bmc4_proxsamadamw_pl1e-2_rho0", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 0}),
    deep_replace(bmc4_proxsamadamw_dummy, {"name": "bmc4_proxsamadamw_pl0_rho2e-3", "optimizer.prox_lambda": 0, "optimizer.rho": 2e-3}),
    deep_replace(bmc4_proxsamadaptive_dummy, {"name": "bmc4_proxsamadaptive_pl1e-2_rho2e-3", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 2e-3}),
    deep_replace(bmc4_proxsamadaptive_dummy, {"name": "bmc4_proxsamadaptive_pl1e-2_rho0", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 0}),
    deep_replace(bmc4_proxsamadaptive_dummy, {"name": "bmc4_proxsamadaptive_pl0_rho2e-3", "optimizer.prox_lambda": 0, "optimizer.rho": 2e-3}),
    deep_replace(
        bmc4_proxsam_dummy,
        {
            "name": "bmc4_pl1e-2_rho2e-3_lrx2",
            "optimizer.prox_lambda": 1e-2,
            "optimizer.rho": 2e-3,
            "optimizer.lr": 4e-5,
        },
    ),
    deep_replace(
        bmc4_proxsam_dummy,
        {
            "name": "bmc4_pl1e-2_rho2e-3_wdv2",
            "optimizer.prox_lambda": 1e-2,
            "optimizer.rho": 2e-3,
            "optimizer.weight_decay": 5e-2,
        },
    ),
    deep_replace(
        bmc4_proxsam_dummy,
        {
            "name": "bmc4_pl1e-2_rho0_lrx2",
            "optimizer.prox_lambda": 1e-2,
            "optimizer.rho": 0,
            "optimizer.lr": 4e-5,
        },
    ),
    deep_replace(
        bmc4_proxsam_dummy,
        {
            "name": "bmc4_pl0_rho2e-3_lrx2",
            "optimizer.prox_lambda": 0,
            "optimizer.rho": 2e-3,
            "optimizer.lr": 4e-5,
        },
    ),
    deep_replace(
        bmc4_proxsam_dummy,
        {
            "name": "bmc4_pl0_rho0_lrx2",
            "optimizer.prox_lambda": 0,
            "optimizer.rho": 0,
            "optimizer.lr": 4e-5,
        },
    ),
    deep_replace(bmc4_proxsamadamw_dummy, {"name": "bmc4_proxsamadamw_pl1e-2_rho2e-3_lrx2", "optimizer.prox_lambda": 1e-2, "optimizer.rho": 2e-3, "optimizer.lr": 4e-5}),

)
