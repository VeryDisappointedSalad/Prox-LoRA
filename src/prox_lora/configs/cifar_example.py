from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.cifar import CIFAR10Config
from prox_lora.infrastructure.configs import register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.example_cnn import ExampleCNNConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

import prox_lora.optimizers.ISTA_optimizer
import prox_lora.optimizers.FISTA_optimizer
import prox_lora.optimizers.ADMM_optimizer

register_configs(
    FullTrainConfig(
        name="cifar_example",
        datamodule=CIFAR10Config(augmentations = True),
        dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
        model=ExampleCNNConfig(input_shape=(3, 32, 32), hidden_channels=(120, 84), num_classes=10),
        #optimizer=OptimizerConfig(opt="adamw", lr=0.002, weight_decay=1e-5, momentum=0.9),
        optimizer=OptimizerConfig(opt="admm", lr=0.002, weight_decay=1e-5, momentum=0.9),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=20, warmup_epochs=1, min_lr=1e-5),
        trainer=TrainerConfig(max_epochs=10, log_every_n_steps=50),
    )
)
