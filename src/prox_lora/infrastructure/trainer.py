import pprint
from dataclasses import asdict, dataclass, field
from pathlib import Path

import clearml
import lightning as L
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT_STR
from lightning.pytorch.callbacks import (
    DeviceStatsMonitor,
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger

from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.datasets.cifar import CIFAR10Config
from prox_lora.datasets.mnist import MNISTConfig
from prox_lora.infrastructure.configs import deep_asdict, save_config, yaml
from prox_lora.models.classifier import Classifier
from prox_lora.models.example_cnn import ExampleCNNConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig
from prox_lora.utils.io import PROJECT_ROOT


@yaml.register_class
@dataclass(frozen=True)
class TrainerConfig:
    """
    Configuration for `lightning.Trainer`.

    See: https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.trainer.trainer.Trainer.html
    """

    precision: _PRECISION_INPUT_STR = "32-true"
    accumulate_grad_batches: int = 1
    max_epochs: int = 10
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    limit_test_batches: int | None = None
    val_check_interval: float = 1.0
    log_every_n_steps: int = 50
    deterministic: bool = False
    benchmark: bool = False
    detect_anomaly: bool = False


@yaml.register_class
@dataclass(frozen=True)
class FullTrainConfig:
    name: str
    datamodule: MNISTConfig | CIFAR10Config
    model: ExampleCNNConfig
    dataloader: DataLoaderConfig = DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(opt="adamw", lr=0.01, weight_decay=1e-4, momentum=0.9)
    )
    scheduler: SchedulerConfig = SchedulerConfig(sched="none")
    trainer: TrainerConfig = TrainerConfig()
    resume: bool = False


def run_training(
    config: FullTrainConfig,
    model_summary_depth: int = 3,
    metric_for_checkpointing: str = "_loss/val",
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints",
    checkpoint_filename_pattern: str = "epoch={epoch:0>3}",
) -> None:
    pprint.pp(config, indent=4, width=200, depth=4, compact=True)

    datamodule = config.datamodule.instantiate(dataloader=config.dataloader)
    model = config.model.instantiate()
    classifier = Classifier(model=model, optimizer=config.optimizer, scheduler=config.scheduler)

    task: clearml.Task = clearml.Task.init(project_name="Prox-LoRA", task_name=config.name)
    task.set_parameters_as_dict(deep_asdict(config))

    trainer = L.Trainer(
        default_root_dir=checkpoint_dir / config.name,
        **asdict(config.trainer),
        logger=TensorBoardLogger(save_dir=checkpoint_dir, name=config.name, default_hp_metric=False),
        callbacks=[
            DeviceStatsMonitor(cpu_stats=False),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                filename=checkpoint_filename_pattern,
                auto_insert_metric_name=False,
                monitor=metric_for_checkpointing,
                save_top_k=1,
                verbose=False,
            ),
            RichProgressBar(leave=True, console_kwargs=dict(force_terminal=True, force_interactive=True, width=250)),
            RichModelSummary(max_depth=model_summary_depth),
        ],
        enable_model_summary=False,  # Disable default model summary in favor of RichModelSummary.
    )

    if trainer.logger:
        assert trainer.logger.log_dir is not None
        version_dir = Path(trainer.logger.log_dir)
        version_dir.mkdir(parents=True, exist_ok=True)
        save_config(config, version_dir / "config.yaml")
    if config.resume:
        trainer.fit(classifier, datamodule, ckpt_path="last")
    else:
        trainer.fit(classifier, datamodule)
