from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.example_cnn import ExampleCNNConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

baseline = FullTrainConfig(
    name="simple_CNN_DR_sgd",
    datamodule=DRConfig(augmentations=True, size=224),
    dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
    model=ExampleCNNConfig(input_shape=(3, 224, 224), hidden_channels=(120, 84), num_classes=5),
    optimizer=OptimizerConfig(opt="sgd", lr=1e-5, weight_decay=1e-5, momentum=0.9),
    scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
    trainer=TrainerConfig(max_epochs=20, log_every_n_steps=50),
)

register_configs(
    baseline,
    deep_replace(
        baseline,
        {
            "name": "simple_CNN_DR_AdamW",
            "optimizer": OptimizerConfig(opt="adamw", lr=0.01, weight_decay=1e-4, momentum=0.9),
        },
    ),
    deep_replace(baseline, {"name": "simple_CNN_DR_ISTA", "optimizer.opt": "ista", "optimizer.prox_lambda": 0.01}),
    deep_replace(baseline, {"name": "simple_CNN_DR_FISTA", "optimizer.opt": "fista", "optimizer.prox_lambda": 0.01}),
    deep_replace(
        baseline,
        {
            "name": "simple_CNN_DR_ADMM",
            "optimizer": OptimizerConfig(
                opt="admm", lr=0.01, weight_decay=1e-4, momentum=0.9, prox_lambda=0.01, rho=0.001
            ),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "simple_CNN_DR_ProxAdam",
            "optimizer": OptimizerConfig(
                opt="proxadam",
                lr=0.01,
                betas=(0.9, 0.999),
                weight_decay=1e-4,
                momentum=0.9,
                prox_lambda=0.01,
                rho=0.001,
            ),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "simple_CNN_DR_ProxSAM",
            "optimizer": OptimizerConfig(
                opt="proxsam", lr=0.01, weight_decay=1e-4, momentum=0.8, prox_lambda=0.01, rho=0.001
            ),
        },
    ),
)
