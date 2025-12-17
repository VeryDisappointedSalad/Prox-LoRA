from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.mnist import MNISTConfig
from prox_lora.infrastructure.configs import register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.example_cnn import ExampleCNNConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

# the basic MNIST example configuration

register_configs(
    FullTrainConfig(
        name="mnist_example",
        datamodule=MNISTConfig(),
        dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
        model=ExampleCNNConfig(input_shape=(3, 28, 28), num_classes=10),
        optimizer=OptimizerConfig(opt="sgd", lr=0.01, weight_decay=1e-4, momentum=0.9),
        scheduler=SchedulerConfig(sched="cosine", num_epochs=10, warmup_epochs=1, min_lr=1e-5),
        # na początku zmniejszyć LR, na początku wyłączyć scheduler
        trainer=TrainerConfig(max_epochs=10, log_every_n_steps=50),
    )
)
