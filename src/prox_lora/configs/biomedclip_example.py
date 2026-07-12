from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.infrastructure.configs import deep_replace, register_configs
from prox_lora.infrastructure.trainer import FullTrainConfig, TrainerConfig
from prox_lora.models.biomedclip import BiomedCLIPConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig

EPOCHS = 10

baseline = FullTrainConfig(
    name="biomedclip_example",
    datamodule=DRConfig(augmentations=True, size=224),
    dataloader=DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True),
    # unfrozen_groups=14 cały model, unfrozen_groups=0 tylko głowa
    model=BiomedCLIPConfig(input_shape=(3, 224, 224), num_classes=5, unfrozen_groups=14, freeze_bn_stats=False),
    optimizer=OptimizerConfig(opt="sgd", lr=1e-5, weight_decay=1e-5, momentum=0.9),
    scheduler=SchedulerConfig(sched="cosine", num_epochs=EPOCHS, warmup_epochs=1, min_lr=1e-5, step_on_epochs=False),
    trainer=TrainerConfig(max_epochs=EPOCHS, log_every_n_steps=50),
)


register_configs(
    baseline,
    # --------------------------------------------------#
    # Standard Gradient ones
    # --------------------------------------------------#
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_AdamW",
            "optimizer": OptimizerConfig(
                opt="adamw",
                lr=1e-2,  # for the head only  1e-3 seems the best for just head tuning, unfrozen_groups=0, 1e-2 za dużo. 1e-4 za mało
                # for the entire thing 1e-4
                weight_decay=1e-4,  # default
                momentum=0.9,  # same
            ),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_AdamW_head_only",
            "optimizer": OptimizerConfig(opt="adamw", lr=1e-3, momentum=0.9),
            "model.unfrozen_groups": 0,
        },
    ),
    deep_replace(
        # lr 1e-5 wd 1e-4 mom 0.9 --> overfitting
        baseline,
        {
            "name": "biomedclip_dr_AdamW_entire_model",
            "optimizer": OptimizerConfig(
                opt="adamw",
                lr=1e-5,  # OKAY
                weight_decay=1e-4,
                momentum=0.9,
            ),
            "model.unfrozen_groups": 14,
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_SGD_head_only",
            "optimizer": OptimizerConfig(
                opt="sgd",
                lr=1e-4,  # OKAY
                weight_decay=1e-4,
                momentum=0.9,
            ),
            "model.unfrozen_groups": 0,
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_SGD_entire_model",
            "optimizer": OptimizerConfig(
                opt="sgd",
                lr=1e-4,  # OKAY
                weight_decay=1e-4,
                momentum=0.9,
            ),
            "model.unfrozen_groups": 14,
        },
    ),
    # --------------------------------------------------#
    # The prox ones #
    # --------------------------------------------------#
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_proxsam_adaptive_entire",
            "optimizer": OptimizerConfig(
                opt="proxsamadaptive", lr=1e-4, weight_decay=1e-4, momentum=0.9, prox_lambda=1e-5, rho=0.01
            ),
            "model.unfrozen_groups": 14,
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_proxsam_adaptive_head",
            "optimizer": OptimizerConfig(
                opt="proxsamadaptive", lr=1e-4, weight_decay=1e-4, momentum=0.9, prox_lambda=1e-5, rho=0.01
            ),
            "model.unfrozen_groups": 0,
        },
    ),
    # --------------------------------------------------#
    # GD
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_proxsam_gd",
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
            "name": "biomedclip_dr_SAM",
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
            "name": "biomedclip_dr_proxsam_adamw_base",
            "optimizer": OptimizerConfig(opt="proxsamadw", lr=1e-3, weight_decay=0, prox_lambda=0, rho=0),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_proxsam_adamw",
            "optimizer": OptimizerConfig(opt="proxsamadw", lr=1e-3, weight_decay=1e-4, prox_lambda=1e-2, rho=0.005),
        },
    ),
    deep_replace(
        baseline,
        {
            "name": "biomedclip_dr_sam_adamw",
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
            "name": "biomedclip_dr_ista",
            "optimizer": OptimizerConfig(opt="ista", lr=1e-3, momentum=0.9, weight_decay=1e-4, prox_lambda=1e-3),
        },
    ),
)
