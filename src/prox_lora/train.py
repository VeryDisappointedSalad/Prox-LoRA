from prox_lora.infrastructure.cli import CLI
from prox_lora.infrastructure.configs import deep_replace, get_config
from prox_lora.infrastructure.trainer import FullTrainConfig, run_training
from prox_lora.utils.io import PROJECT_ROOT


def main() -> None:
    config = get_config(FullTrainConfig, "mnist_example")
    config = deep_replace(config, {"name": "test13"})
    run_training(config, checkpoint_dir=PROJECT_ROOT / "checkpoints")


if __name__ == "__main__":
    CLI(main, seed=42).run()
