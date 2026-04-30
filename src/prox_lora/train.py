"""Run with `uv run src/prox_lora/train.py --help`."""

from pathlib import Path

import tyro

from prox_lora.infrastructure.cli import CLI
from prox_lora.infrastructure.configs import deep_replace, get_config, load_config, save_config
from prox_lora.infrastructure.slurm import SlurmConfig, submit_slurm_job
from prox_lora.infrastructure.trainer import FullTrainConfig, get_new_run_dir, run_training
from prox_lora.utils.io import PROJECT_ROOT

DEFAULT_SLURM_CONFIG = SlurmConfig(duration="00:10:00", partition="common", gpus=1)


def main() -> None:
    tyro.extras.subcommand_cli_from_dict(
        {
            "example": example,
            "start": start_training,
            "resume": resume_training,
            "submit": submit_training,
            "submit-resume": submit_resume_training,
        }
    )


def example() -> None:
    start_training(
        "mnist_example_ISTA",
        {
            # "name": "mnist-april",
            "dataloader.pin_memory": False
            # "clearml_project": None,
        },
    )


def start_training(config: str, /, replace: dict[str, bool | int | float | str] | None = None) -> None:
    """
    Run a new training.

    Example:
    ```
        train.py start mnist_example_ISTA --replace dataloader.pin_memory False optimizer.lr 0.01
    ```

    Args:
        config: Name of a registered FullTrainConfig, like "mnist_example_ISTA".
        replace: Quick replacements for FullTrainConfig values.
    """
    cfg = get_config(FullTrainConfig, config)
    cfg = deep_replace(cfg, replace or {})
    run_dir = get_new_run_dir(PROJECT_ROOT / "runs" / cfg.name)
    save_config(cfg, run_dir / "config.yaml")
    run_training(cfg, run_dir=run_dir, resume=False)
    print("Training finished.")


def resume_training(run_dir: Path, /) -> None:
    """Resume training from a run directory containing a config.yaml."""
    if not (run_dir / "config.yaml").exists():
        raise ValueError(f"Run directory {run_dir} should contain config.yaml.")
    cfg: FullTrainConfig = load_config(run_dir / "config.yaml")
    resume = (run_dir / "checkpoints").exists()
    run_training(cfg, run_dir=run_dir.absolute(), resume=resume)
    print("Training finished.")


def submit_training(
    config: str,
    /,
    replace: dict[str, bool | int | float | str] | None = None,
    slurm: SlurmConfig = DEFAULT_SLURM_CONFIG,
    *,
    follow: bool = False,
) -> None:
    """Submit a new training job via SLURM."""
    cfg = get_config(FullTrainConfig, config)
    cfg = deep_replace(cfg, replace or {})

    run_dir = get_new_run_dir(PROJECT_ROOT / "runs" / cfg.name)
    save_config(cfg, run_dir / "config.yaml")

    submit_slurm_job(
        slurm,
        job_name=cfg.name + "/" + run_dir.name,
        run_dir=run_dir,
        job_args=["uv", "run", PROJECT_ROOT / "src" / "prox_lora" / "train.py", "resume", run_dir],
        follow=follow,
    )


def submit_resume_training(
    run_dir: Path, /, slurm: SlurmConfig = DEFAULT_SLURM_CONFIG, *, follow: bool = False
) -> None:
    """Submit a SLURM job to resume some training."""
    submit_slurm_job(
        slurm,
        job_name=run_dir.parent.name + "/" + run_dir.name,
        run_dir=run_dir,
        job_args=["uv", "run", PROJECT_ROOT / "src" / "prox_lora" / "train.py", "resume", run_dir],
        follow=follow,
    )


if __name__ == "__main__":
    CLI(main).run()
