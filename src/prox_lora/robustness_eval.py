import json
from pathlib import Path

import foolbox as fb
import matplotlib.pyplot as plt
import torch
import tyro
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from tqdm import tqdm

from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.infrastructure.configs import load_config
from prox_lora.infrastructure.trainer import FullTrainConfig
from prox_lora.models.classifier import Classifier
from prox_lora.utils.io import PROJECT_ROOT


def find_latest_checkpoint(base_run_dir: Path) -> Path | None:
    if not base_run_dir.exists():
        return None
    checkpoints = list(base_run_dir.glob("**/checkpoints/last.ckpt"))
    if not checkpoints:
        checkpoints = list(base_run_dir.glob("**/*.ckpt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0]


def run_adversarial_eval(
    checkpoint_path: str, config: FullTrainConfig, target_count: int, test_batch_size: int, device: str = "cuda"
) -> tuple[list[float], list[float]]:

    actual_device = device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: Requested 'cuda' but GPU is not available! Falling back to 'cpu'. This will be very slow.")
        actual_device = "cpu"

    model_instance = config.model.instantiate()
    model_module = Classifier.load_from_checkpoint(
        checkpoint_path,
        model=model_instance,
        num_classes=config.model.num_classes,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
    )
    model = model_module.model.to(actual_device).eval()

    eval_dataloader_cfg = DataLoaderConfig(
        batch_size=test_batch_size, num_workers=4, pin_memory=(actual_device == "cuda")
    )
    datamodule = config.datamodule.instantiate(dataloader=eval_dataloader_cfg)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    mean = list(OPENAI_DATASET_MEAN)
    std = list(OPENAI_DATASET_STD)
    preprocessing = dict(mean=mean, std=std, axis=-3)

    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=actual_device, preprocessing=preprocessing)
    attack = fb.attacks.LinfPGD()
    epsilons = [0.0, 0.0001, 0.001, 0.01, 0.02]

    total_success = []
    current_count = 0

    print(f"PGD Attack on: {checkpoint_path} (Device: {actual_device})")
    mean_t = torch.tensor(mean).view(3, 1, 1).to(actual_device)
    std_t = torch.tensor(std).view(3, 1, 1).to(actual_device)

    _total_batches = min(len(test_loader), (target_count + test_batch_size - 1) // test_batch_size)

    with tqdm(total=target_count, desc="Evaluating Images") as pbar:
        for _batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(actual_device), labels.to(actual_device)

            unnormalized = (images * std_t + mean_t).clamp(0, 1)

            _, _, success = attack(fmodel, unnormalized, labels, epsilons=epsilons)

            total_success.append(success.cpu())

            # Update progress
            batch_size = images.size(0)
            added_count = min(batch_size, target_count - current_count)
            current_count += added_count
            pbar.update(added_count)

            if current_count >= target_count:
                break

    print(f"\nEvaluated {min(current_count, target_count)} images successfully.")

    combined_success = torch.cat(total_success, dim=-1)[:, :target_count]
    robust_accuracy = 1.0 - combined_success.float().mean(axis=-1).numpy()

    del model, fmodel, model_module
    torch.cuda.empty_cache()

    return epsilons, robust_accuracy.tolist()


def plot_robustness_curves(results_dict: dict, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    plt.figure(figsize=(10, 6))

    for model_name, (eps, acc) in results_dict.items():
        plt.plot(eps, acc, marker="o", label=model_name)

    plt.title(r"Robustness Curve: Accuracy vs. Adversarial Noise ($\epsilon$)")
    plt.xlabel(r"Perturbation Magnitude ($\epsilon$)")
    plt.ylabel("Accuracy")
    plt.grid(visible=True, linestyle="--")
    plt.legend()

    plot_path = output_dir / "robustness_curve.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot dynamically updated: {plot_path}")


def save_results_json(results_dict: dict, output_dir: Path):
    json_path = output_dir / "robustness_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)


def main(
    target_count: int = 64,
    test_batch_size: int = 16,
    output_dir: str = "plots/robustness",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:

    print("Starting Robustness Evaluation Script...")
    out_path = PROJECT_ROOT / output_dir

    runs_root = PROJECT_ROOT / "runs"
    model_directories = {
        "AdamW_head": runs_root / "biomedclip_dr_AdamW_head_only",
        # "SGD_head": runs_root / "biomedclip_dr_SGD_head_only",
        "AdamW_entire": runs_root / "biomedclip_dr_AdamW_entire_model",
        # "SGD_entire": runs_root / "biomedclip_dr_SGD_entire_model",
        "ConvNetAdamW": runs_root / "convnet_dr_AdamW",
        "ProxSamAdaptive_entire": runs_root / "biomedclip_dr_proxsam_adaptive_entire",
        "ProxSamAdaptive_head": runs_root / "biomedclip_dr_proxsam_adaptive_head",
    }

    checkpoints = {}
    for name, dir_path in model_directories.items():
        latest_ckpt = find_latest_checkpoint(dir_path)  # Upewnij się, że dokopiowałeś tę funkcję na górę pliku
        if latest_ckpt:
            checkpoints[name] = latest_ckpt

        all_results = {}

    json_path = out_path / "robustness_results.json"
    if json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
            print(f"Loaded existing partial results for: {list(all_results.keys())}")

    for name, ckpt_rel_path in checkpoints.items():
        if name in all_results:
            print(f"Skipping {name}, already evaluated.")
            continue

        full_ckpt_path = PROJECT_ROOT / ckpt_rel_path
        config_path = full_ckpt_path.parent.parent / "config.yaml"

        if full_ckpt_path.exists() and config_path.exists():
            print(f"\nEntering into {name}...")

            run_config = load_config(config_path)

            eps, acc = run_adversarial_eval(
                checkpoint_path=str(full_ckpt_path),
                config=run_config,
                target_count=target_count,
                test_batch_size=test_batch_size,
                device=device,
            )
            all_results[name] = (eps, acc)

            plot_robustness_curves(all_results, out_path)
            save_results_json(all_results, out_path)
        else:
            print(f"Skipping {name}, checkpoint or config.yaml not found!")

    print("\nAll models evaluated successfully!")


if __name__ == "__main__":
    tyro.cli(main)
