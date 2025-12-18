from prox_lora.infrastructure.cli import CLI
from prox_lora.infrastructure.configs import deep_replace, get_config
from prox_lora.infrastructure.trainer import FullTrainConfig, run_training
from prox_lora.utils.io import PROJECT_ROOT


def main() -> None:
    #config = get_config(FullTrainConfig, "mnist_example_FISTA")
    #config = deep_replace(config, {"name": "FISTA_mnist_after_LR_and_config_debug"})
    config = get_config(FullTrainConfig, "cifar_example")
    config = deep_replace(config, {"name": "cifar-admm-rho=0.001", "dataloader.pin_memory": False})
    run_training(config, checkpoint_dir=PROJECT_ROOT / "checkpoints")


if __name__ == "__main__":
    CLI(main, seed=42).run()
