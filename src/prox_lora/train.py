"""Run with `uv run src/prox_lora/train.py --help`."""

from pathlib import Path

import tyro

from prox_lora.infrastructure.cli import CLI
from prox_lora.infrastructure.configs import deep_replace, get_config, load_config, save_config
from prox_lora.infrastructure.slurm import SlurmConfig, submit_slurm_job
from prox_lora.infrastructure.trainer import FullTrainConfig, get_new_run_dir, run_training
from prox_lora.utils.io import PROJECT_ROOT

DEFAULT_SLURM_CONFIG = SlurmConfig(
    duration="06:00:00",
    partition="h100",
    gpus=1,
    cpus=8,
    mem="30G",  # 32 cpu 128G is proportional,  8cpu 96G by default (weirdly low cpu)?.
)

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
    # start_training(
    #     "mnist_example_ISTA",
    #     {
    #         # "name": "mnist-april",
    #         "dataloader.pin_memory": False,
    #         "wandb_project": "test",
    #     },
    # )

    # for s in [
    #     # "biomedclip_example",
    #     # "biomedclip_dr_AdamW_head_only",
    #     # "biomedclip_dr_AdamW_entire_model",
    #     # "biomedclip_dr_SGD_head_only",
    #     # "biomedclip_dr_SGD_entire_model",
    #     # "biomedclip_dr_proxsam_adaptive_entire",
    #     # "biomedclip_dr_proxsam_adaptive_entire_zero_rho",
    #     # "biomedclip_dr_proxsam_adaptive_entire_zero_proxlambda",
    #     # "biomedclip_dr_proxsam_adaptive_head",
    #     # "biomedclip_dr_proxsam_gd", # FAILED with sam_closure, re-ran with closure=sam_closure from here on. And float32 everywhere.
    #     # "biomedclip_dr_SAM",
    #     # "biomedclip_dr_proxsam_adamw_base",
    #     # "biomedclip_dr_proxsam_adamw",
    #     "biomedclip_dr_sam_adamw",
    #     "biomedclip_dr_ista",
    # ]:
    #     start_training(s, {"trainer.precision": "32-true"})

    # for s in [
    #     "conv2_proxsamadw",
    #     "conv2_proxsamadw_nowd",
    # ]:
    #     submit_training(s)

    for prox_lambda in (0,): #  1e-4, 1e-3):
        # for lr in (5e-2, 2e-2, 5e-3):
        for rho in (0.05, 0.01, 0.1, 0.02, 0.2):
            #  0.01, 0.02, 0.05, 0.1, 0.2 # 0.1 bad, 0.05 okay, 0.01 też okay ale gorsze niż 0.05
            submit_training(
                "conv2_proxsamadw",
                {

                    "name": f"conv2_proxsamadw_pl{_str(prox_lambda)}_rho{_str(rho)}",
                    "optimizer.opt": "ista",
                    # "optimizer.lr": lr,
                    "optimizer.prox_lambda": prox_lambda,
                    "optimizer.rho": rho,
                }
            )

def _str(x: float) -> str:
    """Format 0.03 as 3e-2."""
    if x == 0:
        return "0"
    exponent = int(f"{x:e}".split("e")[-1])
    mantissa = x / (10**exponent)
    mantissa = round(mantissa * 10**3) / 10**3
    if abs(mantissa - int(mantissa)) < 1e-3:
        mantissa = int(mantissa)
    return f"{mantissa}e{exponent}"

    # for s in (112, 64):
    #     for batch_size in (128, 64, 256):
    #         for lr in (1e-1, 2e-1, 5e-2):
    #             start_training(
    #                 "cifar_timm3_sgd",
    #                 {
    #                     "name": f"cifar_timm3_sgd_s{s}_b{batch_size}_lr{lr}",
    #                     "datamodule.target_image_size": s,
    #                     "dataloader.batch_size": batch_size,
    #                     "optimizer.lr": lr,
    #                 },
    #             )


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
