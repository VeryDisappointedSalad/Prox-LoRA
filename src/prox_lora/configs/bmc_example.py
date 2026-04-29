from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.cifar import CIFAR10Config
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.biomedclip import BiomedCLIPConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

register_configs(
    FullTrainConfig(
        name="bmc_example",
        datamodule=CIFAR10Config(augmentations=True),
        dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
        model=BiomedCLIPConfig(input_shape=(3, 32, 32), num_classes=10, unfrozen_groups=2, freeze_bn_stats=False),
        optimizer=OptimizerConfig(opt="adamw", lr=2e-3, weight_decay=1e-5, momentum=0.9),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=20, log_every_n_steps=50),
    ),
    FullTrainConfig(
        name="dr_bmc",
        datamodule=DRConfig(augmentations=True, size=224),
        dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
        model=BiomedCLIPConfig(input_shape=(3, 224, 224), num_classes=5, unfrozen_groups=2, freeze_bn_stats=False),
        optimizer=OptimizerConfig(opt="adamw", lr=1e-5, weight_decay=1e-5, momentum=0.9),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=20, log_every_n_steps=50),
    ),
)
