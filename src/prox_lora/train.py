from prox_lora.infrastructure.cli import CLI
from prox_lora.infrastructure.configs import deep_replace, get_config
from prox_lora.infrastructure.trainer import FullTrainConfig, run_training
from prox_lora.utils.io import PROJECT_ROOT


def main() -> None:
    #config = get_config(FullTrainConfig, "bmc_example")
    config = get_config(FullTrainConfig, "CNN_retinopathy")
    #config = deep_replace(config, {"name": "BioMed_DR_Adam", "dataloader.pin_memory": False})
    config = deep_replace(config, {"name": "CNN_DR_AdamW", "dataloader.pin_memory": False})
    run_training(config, checkpoint_dir=PROJECT_ROOT / "checkpoints")


if __name__ == "__main__":
    CLI(main, seed=42).run()

