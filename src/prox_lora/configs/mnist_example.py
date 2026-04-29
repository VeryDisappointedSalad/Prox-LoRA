from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.mnist import MNISTConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.example_cnn import ExampleCNNConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

baseline = FullTrainConfig(
    name="mnist_example",
    datamodule=MNISTConfig(),
    dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
    model=ExampleCNNConfig(input_shape=(3, 28, 28), num_classes=10),
    optimizer=OptimizerConfig(opt="adamw", lr=0.01, weight_decay=1e-4, momentum=0.9),
    scheduler=SchedulerConfig(sched="cosine", num_epochs=10, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
    trainer=TrainerConfig(max_epochs=10, log_every_n_steps=50),
)

register_configs(
    baseline,
    deep_replace(baseline, {"name": "mnist_example_ISTA", "optimizer.opt": "ista", "optimizer.prox_lambda": 0.01}),
    deep_replace(baseline, {"name": "mnist_example_FISTA", "optimizer.opt": "fista", "optimizer.prox_lambda": 0.01}),
    deep_replace(
        baseline,
        {
            "name": "mnist_example_ADMM",
            "optimizer": OptimizerConfig(
                opt="admm", lr=0.01, weight_decay=1e-4, momentum=0.9, prox_lambda=0.01, rho=0.001
            ),
        },
    ),
)
