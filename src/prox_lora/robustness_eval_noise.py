import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torchmetrics.classification import Accuracy, CohenKappa, F1Score
from tqdm import tqdm

from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.infrastructure.configs import load_config
from prox_lora.infrastructure.trainer import FullTrainConfig
from prox_lora.models.classifier import Classifier
from prox_lora.utils.io import PROJECT_ROOT


def run_noise_eval(
    checkpoint_path: str, config: FullTrainConfig, target_count: int, test_batch_size: int, device: str = "cuda"
) -> tuple[list[float], dict[str, list[float]]]:

    actual_device = device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: Requested 'cuda' but GPU is not available! Falling back to 'cpu'.")
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
    mean_t = torch.tensor(mean).view(3, 1, 1).to(actual_device)
    std_t = torch.tensor(std).view(3, 1, 1).to(actual_device)

    noise_sigmas = list(np.linspace(0, 0.5, 20, endpoint=True))
    num_classes = config.model.num_classes

    metrics_per_sigma = {
        sigma: {
            "Accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(actual_device),
            "Quadratic_Kappa": CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic").to(
                actual_device
            ),
            "F1_Macro": F1Score(task="multiclass", num_classes=num_classes, average="macro").to(actual_device),
        }
        for sigma in noise_sigmas
    }

    current_count = 0
    print(f"Gaussian Noise Comprehensive Eval on: {checkpoint_path} (Device: {actual_device})")

    with torch.no_grad():
        with tqdm(total=target_count, desc="Evaluating Images") as pbar:
            for _batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(actual_device), labels.to(actual_device)

                unnormalized = (images * std_t + mean_t).clamp(0, 1)
                batch_size = images.size(0)
                added_count = min(batch_size, target_count - current_count)

                unnormalized = unnormalized[:added_count]
                labels = labels[:added_count]

                for sigma in noise_sigmas:
                    if sigma == 0.0:
                        noisy_images = unnormalized
                    else:
                        noise = torch.randn_like(unnormalized) * sigma
                        noisy_images = (unnormalized + noise).clamp(0, 1)

                    normalized_noisy = (noisy_images - mean_t) / std_t
                    logits = model(normalized_noisy)
                    predictions = logits.argmax(dim=-1)

                    metrics_per_sigma[sigma]["Accuracy"].update(predictions, labels)
                    metrics_per_sigma[sigma]["Quadratic_Kappa"].update(predictions, labels)
                    metrics_per_sigma[sigma]["F1_Macro"].update(predictions, labels)

                current_count += added_count
                pbar.update(added_count)
                if current_count >= target_count:
                    break

    print(f"\nEvaluated {current_count} images successfully.")

    model_history = {"Accuracy": [], "Quadratic_Kappa": [], "F1_Macro": []}
    for sigma in noise_sigmas:
        model_history["Accuracy"].append(round(float(metrics_per_sigma[sigma]["Accuracy"].compute().item()), 4))
        model_history["Quadratic_Kappa"].append(
            round(float(metrics_per_sigma[sigma]["Quadratic_Kappa"].compute().item()), 4)
        )
        model_history["F1_Macro"].append(round(float(metrics_per_sigma[sigma]["F1_Macro"].compute().item()), 4))

    del model, model_module
    torch.cuda.empty_cache()

    return noise_sigmas, model_history


def plot_robustness_curves(results_dict: dict, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics_to_plot = ["Accuracy", "Quadratic_Kappa", "F1_Macro"]

    for metric_name in metrics_to_plot:
        plt.figure(figsize=(10, 6))

        for model_name, data in results_dict.items():
            sigmas = data["sigmas"]
            metric_values = data[metric_name]
            plt.plot(sigmas, metric_values, marker="o", linewidth=2, label=model_name)

        plt.title(rf"Robustness Curve: {metric_name.replace('_', ' ')} vs. Gaussian Noise ($\sigma$)")
        plt.xlabel(r"Noise Standard Deviation ($\sigma$)")
        plt.ylabel(metric_name.replace("_", " "))
        plt.grid(visible=True, linestyle="--")
        plt.legend(loc="lower left" if metric_name != "Accuracy" else "upper right")
        plt.tight_layout()

        plot_path = output_dir / f"robustness_{metric_name.lower()}_curve.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated plot: {plot_path}")


def save_results_json(results_dict: dict, output_dir: Path):
    json_path = output_dir / "robustness_noise_results.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)


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


def main(
    target_count: int = 1024,
    test_batch_size: int = 16,
    output_dir: str = "plots/robustness",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:

    print("Starting Comprehensive Gaussian Noise Robustness Script...")
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
        latest_ckpt = find_latest_checkpoint(dir_path)
        if latest_ckpt:
            checkpoints[name] = latest_ckpt
            print(f"🔎 Found checkpoint for {name}: {latest_ckpt.relative_to(PROJECT_ROOT)}")

    all_results = {}
    json_path = out_path / "robustness_noise_results.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                all_results = json.load(f)
                print(f"Loaded existing results for: {list(all_results.keys())}")
        except Exception:
            all_results = {}

    for name, full_ckpt_path in checkpoints.items():
        if name in all_results:
            print(f"Skipping {name}, already evaluated.")
            continue

        config_path = full_ckpt_path.parent.parent / "config.yaml"

        if full_ckpt_path.exists() and config_path.exists():
            print(f"\nEntering into {name}...")
            run_config = load_config(config_path)

            sigmas, history = run_noise_eval(
                checkpoint_path=str(full_ckpt_path),
                config=run_config,
                target_count=target_count,
                test_batch_size=test_batch_size,
                device=device,
            )

            all_results[name] = {
                "sigmas": sigmas,
                "Accuracy": history["Accuracy"],
                "Quadratic_Kappa": history["Quadratic_Kappa"],
                "F1_Macro": history["F1_Macro"],
            }

            plot_robustness_curves(all_results, out_path)
            save_results_json(all_results, out_path)
        else:
            print(f"Skipping {name}, missing config or checkpoint.")

    print("\nAll tasks finalized successfully!")


if __name__ == "__main__":
    tyro.cli(main)
