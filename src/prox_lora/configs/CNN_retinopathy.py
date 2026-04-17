from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.example_cnn import ExampleCNNConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

import prox_lora.optimizers.ISTA_optimizer
import prox_lora.optimizers.FISTA_optimizer
import prox_lora.optimizers.ADMM_optimizer
import prox_lora.optimizers.ProxAdam
import prox_lora.optimizers.AdaProx
import prox_lora.optimizers.ProxSAM

register_configs(
    FullTrainConfig(
        name="CNN_retinopathy",
        datamodule=DRConfig(augmentations = True),
        dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
        model=ExampleCNNConfig(input_shape=(3, 224, 224), num_classes=5),
        optimizer=OptimizerConfig(opt="adamw", lr=0.01, weight_decay=1e-4, momentum=0.5),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=15, warmup_epochs=1, decay_epochs=15, min_lr=1e-5, step_on_epochs=False),
        trainer=TrainerConfig(max_epochs=15, log_every_n_steps=50),
    )
)
