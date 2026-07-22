from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.biomedclip import BiomedCLIPConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

v2_baseline = FullTrainConfig(
    name="v2_baseline",
    datamodule=DRConfig(augmentations=True, size=512, weighted_sampler=False),
    dataloader=DataLoaderConfig(batch_size=32, num_workers=16, pin_memory=True),
    # unfrozen_groups=14 cały model, unfrozen_groups=0 tylko głowa
    model=BiomedCLIPConfig(input_shape=(3, 512, 512), num_classes=5, unfrozen_groups=14, freeze_bn_stats=False),
    optimizer=OptimizerConfig(opt="adamw", lr=1e-6, weight_decay=1e-4, momentum=0.9),
    scheduler=SchedulerConfig(sched="cosine", num_epochs=10, warmup_epochs=1, min_lr=1e-7, step_on_epochs=False),
    trainer=TrainerConfig(
        precision="32-true",
        accumulate_grad_batches=1,  # TODO: throws an error for any > 1
        max_epochs=10,
        log_every_n_steps=100,
    ),
    loss_class_weights_gamma=0.0,
)

register_configs(
    v2_baseline,
    deep_replace(v2_baseline, {"name": "v2_sampler", "datamodule.weighted_sampler": True}),
    deep_replace(v2_baseline, {"name": "v2_samplerSqrt", "datamodule.weighted_sampler": "sqrt"}),
    deep_replace(
        v2_baseline,
        {"name": "v2_sampler_classgamma+1", "datamodule.weighted_sampler": True, "loss_class_weights_gamma": 1.0},
    ),
    deep_replace(
        v2_baseline,
        {"name": "v2_sampler_classgamma+0.5", "datamodule.weighted_sampler": True, "loss_class_weights_gamma": 0.5},
    ),
    deep_replace(
        v2_baseline,
        {"name": "v2_samplerSqrt_classgamma+1", "datamodule.weighted_sampler": "sqrt", "loss_class_weights_gamma": 1.0},
    ),
    deep_replace(
        v2_baseline,
        {
            "name": "v2_samplerSqrt_classgamma+0.5",
            "datamodule.weighted_sampler": "sqrt",
            "loss_class_weights_gamma": 0.5,
        },
    ),
)
