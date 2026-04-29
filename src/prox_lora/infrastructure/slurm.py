import subprocess
from dataclasses import dataclass
from pathlib import Path

from prox_lora.infrastructure.configs import yaml
from prox_lora.utils.io import PROJECT_ROOT


@yaml.register_class
@dataclass(frozen=True)
class SlurmConfig:
    duration: str = "00:10:00"  # Duration in format "DD-HH:MM:SS" or smaller parts like "MM:SS"
    partition: str = "common"
    mem: str | None = None  # like "16G" for GiB.
    cpus: int | None = None
    gpus: int = 1
    mail: str | None = None


def submit_slurm_job(slurm_config: SlurmConfig, job_name: str, log_path: Path, job_args: list[str | Path]) -> None:
    args = [
        "sbatch",
        f"--chdir={PROJECT_ROOT}",
        f"--time={slurm_config.duration}",
        f"--partition={slurm_config.partition}",
        "--ntasks=1",
        f"--job-name={job_name}",
        f"--output={log_path.absolute()}",
        f"--error={log_path.absolute()}",
    ]
    if slurm_config.mem is not None:
        args += ["--mem", slurm_config.mem]
    if slurm_config.cpus is not None:
        args += ["--cpus-per-task", str(slurm_config.cpus)]
    if slurm_config.gpus > 0:
        args += ["--gpus-per-node", str(slurm_config.gpus)]
    if slurm_config.mail is not None:
        args += ["--mail-user", slurm_config.mail, "--mail-type=ALL"]

    args += ["srun"]
    args += [str(arg.absolute()) if isinstance(arg, Path) else arg for arg in job_args]

    r = subprocess.run(args, check=True, capture_output=True)
    job_id = r.stdout.decode().strip()
    print(f"Submitted SLURM job with ID: {job_id}.")
    (log_path.parent / f"slurm_job_{job_id}").touch()
    # TODO start monitoring logs by default.
