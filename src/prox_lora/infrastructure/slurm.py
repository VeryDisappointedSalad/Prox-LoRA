import shlex
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


def submit_slurm_job(
    slurm_config: SlurmConfig, job_name: str, run_dir: Path, job_args: list[str | Path], *, follow: bool = True
) -> None:
    log_path = run_dir / "log.txt"
    script_path = run_dir / "sbatch.sh"

    args = [str(arg.absolute()) if isinstance(arg, Path) else arg for arg in job_args]
    script_path.write_text(make_sbatch_script(slurm_config, job_name, log_path, shlex.join(args)))

    r = subprocess.run(["sbatch", "--parsable", str(script_path)], check=True, capture_output=True)

    job_id = r.stdout.decode().strip()
    print(f"Submitted SLURM job with ID: {job_id}.")
    (run_dir / f"slurm_job_{job_id}").touch()

    if not follow:
        print(f"To follow logs:\n    tail -f {log_path}\nTo cancel (kill) job:\n    scancel {job_id}")
        print(f"To run something within the same allocation:\n    srun --jobid={job_id} --overlap --pty /bin/bash -i")
        return

    try:
        subprocess.run(
            r"""
                while [ ! -f {log_path} ]; do
                    clear
                    echo "Waiting for job" {job_id} "to start before showing logs... (press Ctrl+C to stop following, the job will not be cancelled)"
                    echo "To cancel job:\n    scancel" {job_id} "\n\n"
                    squeue --Format=jobid:10,username:20,name:40,partition:8,submittime,timeleft:16,state:10,reason:20,nodelist,tres-alloc:60 --partition={partition}
                    sleep 1
                done
                clear
                echo "Following logs for job" {job_id} "... (press Ctrl+C to stop following, the job will continue running)"
                echo "To cancel (kill) job:\n    scancel" {job_id}
                echo "To run something within the same allocation:\n    srun --jobid="{job_id}" --overlap --pty /bin/bash -i\n\n"
                tail -f {log_path}
            """.format(
                log_path=shlex.quote(str(log_path.absolute())),
                job_id=shlex.quote(job_id),
                partition=shlex.quote(slurm_config.partition),
            ),
            shell=True,
            check=False,
        )
    except KeyboardInterrupt:
        print(f"\n\n\nTo follow again:\n    tail -f {log_path}\nTo cancel (kill) job:\n    scancel {job_id}")
        return


def make_sbatch_script(slurm_config: SlurmConfig, job_name: str, log_path: Path, job_cmd: str) -> str:
    sbatch_args = [
        f"--chdir={PROJECT_ROOT}",
        f"--time={slurm_config.duration}",
        f"--partition={slurm_config.partition}",
        "--ntasks=1",
        f"--job-name={job_name}",
        f"--output={log_path.absolute()}",
        f"--error={log_path.absolute()}",
    ]
    if slurm_config.mem is not None:
        sbatch_args += ["--mem", slurm_config.mem]
    if slurm_config.cpus is not None:
        sbatch_args += ["--cpus-per-task", str(slurm_config.cpus)]
    if slurm_config.gpus > 0:
        sbatch_args += ["--gpus-per-node", str(slurm_config.gpus)]
    if slurm_config.mail is not None:
        sbatch_args += ["--mail-user", slurm_config.mail, "--mail-type=ALL"]

    script = "#!/bin/bash\n"
    for arg in sbatch_args:
        script += f"#SBATCH {arg}\n"
    script += "\nset -euxo pipefail\n\n"
    script += job_cmd + "\n"
    return script
