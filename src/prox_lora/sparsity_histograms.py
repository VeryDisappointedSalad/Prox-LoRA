from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

from prox_lora.infrastructure.configs import load_config
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


def extract_weights(checkpoint_path: str, config, device="cuda") -> np.ndarray:
    actual_device = device if torch.cuda.is_available() else "cpu"

    model_instance = config.model.instantiate()
    model_module = Classifier.load_from_checkpoint(
        checkpoint_path,
        model=model_instance,
        num_classes=config.model.num_classes,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
    )
    model = model_module.model.to(actual_device).eval()

    weights = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "weight" in name and param.requires_grad:
                weights.append(param.detach().cpu().numpy().flatten())

    del model, model_module
    torch.cuda.empty_cache()

    return np.concatenate(weights)


def plot_and_save_histogram(weights: np.ndarray, model_name: str, output_dir: Path):

    total_params = len(weights)
    exact_zeros = np.sum(weights == 0.0)
    near_zeros = np.sum(np.abs(weights) < 1e-6)

    sparsity_exact = (exact_zeros / total_params) * 100
    sparsity_near = (near_zeros / total_params) * 100

    mean_w = np.mean(weights)
    std_w = np.std(weights)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Weight Distribution: {model_name}", fontsize=18, fontweight="bold")

    ax1.hist(weights, bins=150, color="#1f77b4", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_title("Standard Histogram (Linear Scale)", fontsize=14)
    ax1.set_xlabel("Weight Value", fontsize=12)
    ax1.set_ylabel("Frequency (Count)", fontsize=12)
    ax1.grid(visible=True, linestyle="--", alpha=0.6)

    ax2.hist(weights, bins=150, color="#ff7f0e", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax2.set_yscale("log")
    ax2.set_title("Standard Histogram (Log Scale)", fontsize=14)
    ax2.set_xlabel("Weight Value", fontsize=12)
    ax2.set_ylabel("Frequency (Log Scale)", fontsize=12)
    ax2.grid(visible=True, linestyle="--", alpha=0.6)

    stats_text = (
        f"Total Weights: {total_params:,}\n"
        f"Mean: {mean_w:.4f}  |  Std: {std_w:.4f}\n"
        f"Exact Zeros (w == 0): {exact_zeros:,} ({sparsity_exact:.2f}%)\n"
        f"Near Zeros (|w| < 1e-6): {near_zeros:,} ({sparsity_near:.2f}%)"
    )

    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    ax2.text(
        0.95,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=props,
        family="monospace",
    )

    plt.tight_layout()

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    plot_path = output_dir / f"hist_{safe_name}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"   -> Sparsity (Exact Zeros): {sparsity_exact:.2f}%")


def main(output_dir: str = "plots/histograms") -> None:
    print("Starting Model Weight Distribution Analysis...")

    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(exist_ok=True, parents=True)
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
        else:
            print(f"Missing: {name}")

    for name, full_ckpt_path in checkpoints.items():
        config_path = full_ckpt_path.parent.parent / "config.yaml"

        if full_ckpt_path.exists() and config_path.exists():
            print(f"\nProcessing {name}...")
            run_config = load_config(config_path)

            weights = extract_weights(str(full_ckpt_path), run_config)

            plot_and_save_histogram(weights, name, out_path)

        else:
            print(f"Skipping {name}, missing config.")

    print(f"\nAll histograms saved to {out_path.absolute()}")


if __name__ == "__main__":
    tyro.cli(main)
