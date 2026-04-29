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
from prox_lora.datasets.diabetic_retinopathy import DRConfig
from prox_lora.datasets.mnist import MNISTConfig
from prox_lora.infrastructure.cli import seed_everything
from prox_lora.infrastructure.configs import deep_asdict, yaml
from prox_lora.models.biomedclip import BiomedCLIPConfig
from prox_lora.models.classifier import Classifier
from prox_lora.models.example_cnn import ExampleCNNConfig
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig


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
    datamodule: MNISTConfig | CIFAR10Config | DRConfig
    model: ExampleCNNConfig | BiomedCLIPConfig
    dataloader: DataLoaderConfig = DataLoaderConfig(batch_size=64, num_workers=4, pin_memory=True)
    optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(opt="adamw", lr=0.01, weight_decay=1e-4, momentum=0.9)
    )
    scheduler: SchedulerConfig = SchedulerConfig(sched="none")
    trainer: TrainerConfig = TrainerConfig()
    clearml_project: str | None = "Prox-LoRA"
    seed: int = 1


def run_training(
    config: FullTrainConfig,
    run_dir: Path,
    model_summary_depth: int = 3,
    metric_for_checkpointing: str = "_loss/val",
    checkpoint_filename_pattern: str = "epoch={epoch:0>3}",
    *,
    resume: bool = False,
) -> None:
    """
    Run training, saving checkpoints and logs under `run_dir`.

    Args:
    - config: the FullTrainConfig.
    - run_dir: must be of the form ".../{config.name}/r{number}/".
    """
    all_runs_dir = run_dir.parent.parent
    assert config.name == run_dir.parent.name, f"{config.name=} ≠ {run_dir.parent.name=}"
    version = run_dir.name
    assert version.startswith("r"), f"Unexpected {version=}"

    print(f"Training ({resume=}) in run_dir: {run_dir}")
    pprint.pp(config, indent=4, width=200, depth=4, compact=True)

    seed_everything(config.seed)

    datamodule = config.datamodule.instantiate(dataloader=config.dataloader)
    datamodule.prepare_data()
    datamodule.setup()
    steps_in_epoch = len(datamodule.train_dataloader())
    model = config.model.instantiate()
    num_classes = config.model.num_classes
    classifier = Classifier(
        model=model,
        num_classes=num_classes,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        steps_in_epoch=steps_in_epoch,
    )

    task: clearml.Task | None = None
    if config.clearml_project is not None:
        task = clearml.Task.init(
            project_name=config.clearml_project, task_name=config.name + "/" + version, continue_last_task=resume
        )
        task.set_parameters_as_dict(deep_asdict(config))

    trainer = L.Trainer(
        default_root_dir=all_runs_dir / config.name,
        **asdict(config.trainer),
        logger=TensorBoardLogger(save_dir=all_runs_dir, name=config.name, version=version, default_hp_metric=False),
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
            # Note that using save_last=True or "link" above would only copy or link the last top_k checkpoint.
            ModelCheckpoint(filename="last", auto_insert_metric_name=False, verbose=False),
            RichProgressBar(leave=True, console_kwargs=dict(force_terminal=True, force_interactive=True, width=250)),
            RichModelSummary(max_depth=model_summary_depth),
        ],
        enable_model_summary=False,  # Disable default model summary in favor of RichModelSummary.
    )

    try:
        trainer.fit(classifier, datamodule, ckpt_path="last" if resume else None)
    finally:
        if task is not None:
            print("Flushing ClearML, this may take a while...")
            task.flush(wait_for_uploads=True)
            print("Flushed.")
            # task.close()


def get_new_run_dir(d: Path) -> Path:
    """Create a new run directory under `d`, like "r0" or "r1", with an auto-incremented version number."""
    d.mkdir(parents=True, exist_ok=True)
    versions = []
    for p in d.iterdir():
        if p.name.startswith("r"):
            try:
                versions.append(int(p.name[1:]))
            except ValueError:  # Ignore directories that don't follow the "r{number}" pattern.
                pass
    next = max(versions, default=0) + 1
    run_dir = d / f"r{next}"
    try:
        run_dir.mkdir()
    except FileExistsError:
        raise ValueError(f"Run directory {run_dir} already exists (race condition?)") from None
    return run_dir
