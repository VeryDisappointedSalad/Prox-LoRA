from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torchmetrics.classification import MulticlassAccuracy, MulticlassCohenKappa, MulticlassF1Score
from tqdm import tqdm

from prox_lora.datasets.common import SizedDataset
from prox_lora.infrastructure.cli import CLI
from prox_lora.utils.eval import GROUPS, get_checkpoints_to_plot, load_for_eval
from prox_lora.utils.io import PROJECT_ROOT, load_json, save_json_atomic


def run_noise_eval(
    checkpoint_path: Path, target_count: int | None, test_batch_size: int, device: str = "cuda"
) -> tuple[list[float], dict[str, list[float]]]:
    config, model, test_loader = load_for_eval(Path(checkpoint_path), test_batch_size, device)

    if target_count is None:
        target_count = len(cast(SizedDataset[tuple[torch.Tensor, int]], test_loader.dataset))

    mean_t = torch.tensor(list(OPENAI_DATASET_MEAN)).view(3, 1, 1).to(device)
    std_t = torch.tensor(list(OPENAI_DATASET_STD)).view(3, 1, 1).to(device)

    noise_sigmas = list(np.linspace(0, 0.2, 20, endpoint=True))
    num_classes = config.model.num_classes

    metrics_per_sigma = {
        sigma: {
            "Accuracy": MulticlassAccuracy(num_classes=num_classes, average="macro").to(device),
            "Quadratic_Kappa": MulticlassCohenKappa(num_classes=num_classes, weights="quadratic").to(device),
            "F1_Macro": MulticlassF1Score(num_classes=num_classes, average="macro").to(device),
        }
        for sigma in noise_sigmas
    }

    current_count = 0
    print(f"Gaussian Noise Comprehensive Eval on: {checkpoint_path} (Device: {device})")

    with torch.no_grad():
        with tqdm(total=target_count, desc="Evaluating Images") as pbar:
            for _batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)

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
                pbar.set_postfix({"Acc": round(float(metrics_per_sigma[0.0]["Accuracy"].compute().item()), 4)})
                if current_count >= target_count:
                    break

    print(f"\nEvaluated {current_count} images successfully.")

    model_history: dict[str, list[float]] = {"Accuracy": [], "Quadratic_Kappa": [], "F1_Macro": []}
    for sigma in noise_sigmas:
        model_history["Accuracy"].append(round(float(metrics_per_sigma[sigma]["Accuracy"].compute().item()), 4))
        model_history["Quadratic_Kappa"].append(
            round(float(metrics_per_sigma[sigma]["Quadratic_Kappa"].compute().item()), 4)
        )
        model_history["F1_Macro"].append(round(float(metrics_per_sigma[sigma]["F1_Macro"].compute().item()), 4))

    del model
    torch.cuda.empty_cache()

    return noise_sigmas, model_history


def plot_robustness_curves(results_dict: dict[str, dict[str, list[float]]], output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    metrics_to_plot = ["Accuracy", "Quadratic_Kappa", "F1_Macro"]

    group_to_ax = {"BMC_AdamW": (0, 0), "BMC_SGD": (0, 1), "conv_AdamW": (1, 0), "conv_SGD": (1, 1)}
    label_to_line_style = {"base": "-", "ISTA": ":", "SAM": "--", "ProxSAM": "-."}

    for metric_name in metrics_to_plot:
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex=True, sharey=True)
        fig.tight_layout()
        fig.suptitle(rf"Robustness: {metric_name.replace('_', ' ')} vs Gaussian Noise ($\sigma$)", fontsize=14, y=1.01)

        for group_name, members in GROUPS.items():
            ax = axs[group_to_ax[group_name]]

            for label, ckpt_name in members.items():
                data = results_dict.get(ckpt_name)
                assert data is not None, f"Checkpoint {ckpt_name} not found in results_dict."
                sigmas = data["sigmas"]
                metric_values = data[metric_name]

                ls = label_to_line_style[label.removesuffix("Adapt")]
                ax.plot(sigmas, metric_values, marker=".", linewidth=1, label=label, linestyle=ls)

            # ax.set_title(group_name.replace("_", " with "))
            ax.set_xlabel(r"Noise standard deviation ($\sigma$)")
            ax.set_ylabel(metric_name.replace("_", " "))
            ax.grid(visible=True, linestyle="--")
            ax.legend(loc="upper right", title=group_name)
            ax.set_xlim(0, 0.06)
            if metric_name == "Quadratic_Kappa":
                ax.set_ylim(0.3, 0.75)
            else:
                ax.set_ylim(0.3, 0.55)

        plot_path = output_dir / f"noise_vs_{metric_name.lower()}.png"
        fig.savefig(plot_path, dpi=300)
        plt.close()
        print(f"Generated plot: {plot_path}")


def main(
    target_count: int | None = None,
    test_batch_size: int = 128,
    output_dir: str = "plots/robustness",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print("Starting Comprehensive Gaussian Noise Robustness Script...")
    out_path = PROJECT_ROOT / output_dir

    checkpoints = get_checkpoints_to_plot()

    all_results = dict[str, dict[str, list[float]]]()
    json_path = out_path / "robustness_noise_results.json"
    if json_path.exists():
        all_results = load_json(json_path)
        print(f"Loaded existing results for: {list(all_results.keys())}")

    for name, ckpt_path in checkpoints.items():
        if name in all_results:
            print(f"Skipping {name}, already evaluated.")
            continue

        print(f"\nEntering into {name}...")

        sigmas, history = run_noise_eval(
            checkpoint_path=ckpt_path, target_count=target_count, test_batch_size=test_batch_size, device=device
        )

        all_results[name] = {
            "sigmas": sigmas,
            "Accuracy": history["Accuracy"],
            "Quadratic_Kappa": history["Quadratic_Kappa"],
            "F1_Macro": history["F1_Macro"],
        }

        save_json_atomic(all_results, json_path)

        # Plot only selected checkpoints, in the order given in `checkpoints`.
        plot_robustness_curves(all_results, out_path)

    # Re-plot in case all results were already ready.
    plot_robustness_curves(all_results, out_path)

    print("\nAll tasks finalized successfully!")


if __name__ == "__main__":
    CLI().before_main()
    tyro.cli(main)
