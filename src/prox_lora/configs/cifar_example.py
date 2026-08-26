from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.cifar import CIFAR10Config
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.example_cnn import ExampleCNNConfig
from prox_lora.models.timm import TimmConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

baseline = FullTrainConfig(
    name="cifar_example",
    datamodule=CIFAR10Config(augmentations=True),
    dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
    model=ExampleCNNConfig(input_shape=(3, 32, 32), hidden_channels=(120, 84), num_classes=10),
    optimizer=OptimizerConfig(opt="adamw", lr=2e-3, weight_decay=1e-5, momentum=0.9),
    scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
    trainer=TrainerConfig(max_epochs=10, log_every_n_steps=50),
)

register_configs(
    baseline,
    FullTrainConfig(
        name="cifar_timm",
        datamodule=CIFAR10Config(augmentations=True, target_image_size=224),
        dataloader=DataLoaderConfig(batch_size=128, num_workers=4, pin_memory=True),
        model=TimmConfig(model_name="mobilenetv2_120d", pretrained=False, num_classes=10),
        optimizer=OptimizerConfig(opt="adamw", lr=2e-2, weight_decay=1e-5, momentum=0.9),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=20, log_every_n_steps=50),
    ),
    FullTrainConfig(
        name="cifar_timm3",
        datamodule=CIFAR10Config(augmentations=True, target_image_size=224),
        dataloader=DataLoaderConfig(batch_size=128, num_workers=4, pin_memory=True),
        model=TimmConfig(model_name="mobilenetv3_large_100", pretrained=False, num_classes=10),
        optimizer=OptimizerConfig(opt="adamw", lr=2e-2, weight_decay=1e-5, momentum=0.9),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=20, log_every_n_steps=50),
    ),
    FullTrainConfig(
        name="cifar_timm_sgd",
        datamodule=CIFAR10Config(augmentations=True, target_image_size=224),
        dataloader=DataLoaderConfig(batch_size=128, num_workers=4, pin_memory=True),
        model=TimmConfig(model_name="mobilenetv2_120d", pretrained=False, num_classes=10),
        optimizer=OptimizerConfig(opt="sgd", lr=1e-1, momentum=0.9, weight_decay=4e-5),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-3, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=20, log_every_n_steps=50),
    ),
    FullTrainConfig(
        name="cifar_timm3_sgd",
        datamodule=CIFAR10Config(augmentations=True, target_image_size=224),
        dataloader=DataLoaderConfig(batch_size=128, num_workers=4, pin_memory=True),
        model=TimmConfig(model_name="mobilenetv3_large_100", pretrained=False, num_classes=10),
        optimizer=OptimizerConfig(opt="sgd", lr=1e-1, momentum=0.9, weight_decay=4e-5),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-3, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=20, log_every_n_steps=50),
    ),
    FullTrainConfig(
        name="cifar_timm_sgd_long",
        datamodule=CIFAR10Config(augmentations=True, target_image_size=224),
        dataloader=DataLoaderConfig(batch_size=128, num_workers=4, pin_memory=True),
        model=TimmConfig(model_name="mobilenetv2_120d", pretrained=False, num_classes=10),
        optimizer=OptimizerConfig(opt="sgd", lr=1e-1, momentum=0.9, weight_decay=4e-5),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=350, warmup_epochs=1, min_lr=1e-3, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=350, log_every_n_steps=50),
    ),
    FullTrainConfig(
        name="cifar_timm_sgd_long_noaug",
        datamodule=CIFAR10Config(augmentations=False, target_image_size=224),
        dataloader=DataLoaderConfig(batch_size=128, num_workers=4, pin_memory=True),
        model=TimmConfig(model_name="mobilenetv2_120d", pretrained=False, num_classes=10),
        optimizer=OptimizerConfig(opt="sgd", lr=1e-1, momentum=0.9, weight_decay=4e-5),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=350, warmup_epochs=1, min_lr=1e-3, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=350, log_every_n_steps=50),
    ),
    deep_replace(baseline, {"name": "cifar_example_ISTA", "optimizer.opt": "ista", "optimizer.prox_lambda": 0.01}),
    deep_replace(baseline, {"name": "cifar_example_FISTA", "optimizer.opt": "fista", "optimizer.prox_lambda": 0.01}),
    deep_replace(
        baseline,
        {
            "name": "cifar_example_ADMM",
            "optimizer": OptimizerConfig(
                opt="admm", lr=0.01, weight_decay=1e-4, momentum=0.9, prox_lambda=0.01, rho=0.001
            ),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "cifar_example_ProxAdam",
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
            "name": "cifar_example_ProxSAM",
            "optimizer": OptimizerConfig(
                opt="proxsam", lr=0.01, weight_decay=1e-4, momentum=0.8, prox_lambda=0.01, rho=0.001
            ),
        },
    ),
)
