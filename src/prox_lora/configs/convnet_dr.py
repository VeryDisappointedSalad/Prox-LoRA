from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.ConvNet import KaggleConvNetConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

EPOCHS = 50

baseline = FullTrainConfig(
    name="conv_net_dr",
    datamodule=DRConfig(augmentations=True, size=224),
    dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
    model=KaggleConvNetConfig(input_shape=(3, 224, 224), channels=(32, 64, 128, 256), num_classes=5),
    optimizer=OptimizerConfig(opt="sgd", lr=1e-3, weight_decay=1e-4, momentum=0.9),
    scheduler=SchedulerConfig(sched="cosine", num_epochs=EPOCHS, warmup_epochs=2, min_lr=1e-6, step_on_epochs=False),
    trainer=TrainerConfig(max_epochs=EPOCHS, log_every_n_steps=500),
)

register_configs(
    baseline,
    # --------------------------------------------------#
    # Standard Gradient ones
    # --------------------------------------------------#
    deep_replace(
        baseline,
        {
            "name": "convnet_dr_AdamW",
            "optimizer": OptimizerConfig(opt="adamw", lr=1e-3, weight_decay=1e-4, momentum=0.9),
        },
    ),
    deep_replace(
        baseline,
        {"name": "convnet_dr_SGD", "optimizer": OptimizerConfig(opt="sgd", lr=1e-3, weight_decay=1e-4, momentum=0.9)},
    ),
    # --------------------------------------------------#
    # The prox ones #
    # --------------------------------------------------#
    # --------------------------------------------------#
    # GD
    deep_replace(
        baseline,
        {
            "name": "convnet_dr_proxsam_gd",
            "optimizer": OptimizerConfig(
                opt="proxsam",  # gradient descent
                lr=0.05,
                weight_decay=1e-4,
                momentum=0.9,
                prox_lambda=1e-5,
                rho=0.01,
            ),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "convnet_dr_SAM",
            "optimizer": OptimizerConfig(
                opt="proxsam",
                lr=1e-3,  # 1e-2 bad, 1e-3 okay
                weight_decay=0.0,
                momentum=0.9,
                prox_lambda=0.0,
                rho=0.005,
            ),
        },
    ),
    # --------------------------------------------------#
    # AdamW
    deep_replace(
        baseline,
        {
            "name": "convnet_dr_proxsam_adamw_base",
            "optimizer": OptimizerConfig(opt="proxsamadw", lr=1e-4, weight_decay=0, prox_lambda=0, rho=0),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "convnet_dr_proxsam_adamw",
            "optimizer": OptimizerConfig(opt="proxsamadw", lr=1e-3, weight_decay=1e-4, prox_lambda=1e-2, rho=0.005),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "convnet_dr_sam_adamw",
            "optimizer": OptimizerConfig(
                opt="proxsamadw",
                lr=1e-4,
                weight_decay=0,
                prox_lambda=0,  # 1 bad
                rho=0.02,  # SAM authors do 0.01, 0.02, 0.05, 0.1, 0.2 # 0.1 bad, 0.05 okay, 0.01 też okay ale gorsze niż 0.05
                # TODO: do a histogram of weights and decide the threshold for prox_lambda
            ),
        },
    ),
    # --------------------------------------------------#
    # Basic prox - ISTA, FISTA
    deep_replace(
        baseline,
        {
            "name": "convnet_dr_ista",
            "optimizer": OptimizerConfig(opt="ista", lr=1e-3, momentum=0.9, weight_decay=1e-4, prox_lambda=1e-3),
        },
    ),
)
