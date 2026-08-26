from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.biomedclip import BiomedCLIPConfig
from prox_lora.models.classifier import ContinuousKappaConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

v3_baseline = FullTrainConfig(
    name="v3_baseline",
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
        accumulate_grad_batches=1,  # TODO: throws an error for any > 1
        max_epochs=30,
        log_every_n_steps=100,
    ),
    loss_ce_alpha=0.2,
    loss_class_weights_gamma=0.0,
    continuous_kappa=ContinuousKappaConfig(alpha=5, mu=1.0, use_class_weights=True)
)
v3_b32 = deep_replace(v3_baseline, {"name": "v3_b32", "dataloader.batch_size": 32})



# For SGD, both batch-size 32 and 64 work about the same, best for lr=4e-5.
v3_sgd = deep_replace(
    v3_baseline, {"name": "v3_sgd", "optimizer": OptimizerConfig(opt="sgd", lr=4e-5, weight_decay=1e-4, momentum=0.9)}
)
v3_sgd_b32 = deep_replace(v3_sgd, {"name": "v3_sgd_b32", "dataloader.batch_size": 32})


register_configs(
    v3_baseline,
    deep_replace(v3_b32, {"name": "v3_1024", "datamodule.size": 1024, "model.input_shape": (3, 1024, 1024)}),
    v3_sgd,
    v3_sgd_b32,
    deep_replace(
        v3_baseline, {"name": "v3_nosampler", "datamodule.weighted_sampler": False, "loss_class_weights_gamma": -0.5}
    ),
)
