"""Run with `uv run src/prox_lora/train.py --help`."""

from pathlib import Path

import tyro

from prox_lora.infrastructure.cli import CLI
from prox_lora.infrastructure.configs import deep_replace, get_config, load_config, save_config
from prox_lora.infrastructure.slurm import SlurmConfig, submit_slurm_job
from prox_lora.infrastructure.trainer import FullTrainConfig, get_new_run_dir, run_training
from prox_lora.utils.io import PROJECT_ROOT
from prox_lora.utils.other import format_float

DEFAULT_SLURM_CONFIG = SlurmConfig(duration="06:00:00", partition="h100", gpus=1, cpus=8, mem="30G")


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
    # Start a single training with some replacements:
    # start_training(
    #     "mnist_example_ISTA",
    #     {
    #         # "name": "mnist-april",
    #         "dataloader.pin_memory": False,
    #         "wandb_project": "test",
    #     },
    # )

    # Submit a list of train configs to SLURM.
    # for s in [
    # "biomedclip_dr_proxsam_gd",
    # "biomedclip_dr_SAM",
    # "biomedclip_dr_proxsam_adamw_base",
    # "biomedclip_dr_proxsam_adamw",
    # "biomedclip_dr_sam_adamw",
    # "biomedclip_dr_ista",
    # ]:
    #     submit_training(s, {"trainer.precision": "32-true"})

    # Submit a grid of hyperparameter combinations to SLURM.
    for prox_lambda in (0,):  #  1e-4, 1e-3):
        # for lr in (5e-2, 2e-2, 5e-3):
        for rho in (0.05, 0.01, 0.1, 0.02, 0.2):
            submit_training(
                "conv2_proxsamadw",
                {
                    "name": f"conv2_proxsamadw_pl{format_float(prox_lambda)}_rho{format_float(rho)}",
                    "optimizer.opt": "ista",
                    # "optimizer.lr": lr,
                    "optimizer.prox_lambda": prox_lambda,
                    "optimizer.rho": rho,
                },
            )


def start_training(config: str, /, replace: dict[str, bool | int | float | str | None] | None = None) -> None:
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
