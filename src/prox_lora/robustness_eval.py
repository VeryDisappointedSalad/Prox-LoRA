from pathlib import Path
from typing import cast

import foolbox as fb
import matplotlib.pyplot as plt
import torch
import tyro
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from tqdm import tqdm

from prox_lora.datasets.common import SizedDataset
from prox_lora.infrastructure.cli import CLI
from prox_lora.utils.eval import (
    GROUP_TO_AX,
    GROUPS,
    LABEL_TO_COLOR,
    LABEL_TO_LINESTYLE,
    get_checkpoints_to_plot,
    load_for_eval,
)
from prox_lora.utils.io import PROJECT_ROOT, load_json, save_json_atomic


def run_adversarial_eval(
    checkpoint_path: Path, target_count: int | None, test_batch_size: int, device: str = "cuda"
) -> tuple[list[float], list[float]]:
    _config, model, test_loader = load_for_eval(Path(checkpoint_path), test_batch_size, device)

    if target_count is None:
        target_count = len(cast(SizedDataset[tuple[torch.Tensor, int]], test_loader.dataset))

    mean = list(OPENAI_DATASET_MEAN)
    std = list(OPENAI_DATASET_STD)
    preprocessing = dict(mean=mean, std=std, axis=-3)

    fmodel = fb.models.pytorch.PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    attack = fb.attacks.LinfPGD()
    epsilons = [0.0, 2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4]

    total_success = []
    current_count = 0

    print(f"PGD Attack on: {checkpoint_path} (Device: {device})")
    mean_t = torch.tensor(mean).view(3, 1, 1).to(device)
    std_t = torch.tensor(std).view(3, 1, 1).to(device)

    _total_batches = min(len(test_loader), (target_count + test_batch_size - 1) // test_batch_size)

    with tqdm(total=target_count, desc="Evaluating Images") as pbar:
        for _batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            unnormalized = (images * std_t + mean_t).clamp(0, 1)

            _, _, success = attack(fmodel, unnormalized, labels, epsilons=epsilons)
            # success has shape (len(epsilons), batch_size), dtype bool where True indicates a successful attack.

            total_success.append(success.cpu())

            # Update progress
            batch_size = images.size(0)
            added_count = min(batch_size, target_count - current_count)
            current_count += added_count
            pbar.update(added_count)

            combined_success = torch.cat(total_success, dim=-1)[:, :target_count]
            robust_accuracy = 1.0 - combined_success.float().mean(dim=-1).numpy()
            pbar.set_postfix({f"Acc@{eps}": round(float(robust_accuracy[i]), 4) for i, eps in enumerate(epsilons)})

            if current_count >= target_count:
                break

    print(f"\nEvaluated {min(current_count, target_count)} images successfully.")

    combined_success = torch.cat(total_success, dim=-1)[:, :target_count]
    robust_accuracy = 1.0 - combined_success.float().mean(dim=-1).numpy()

    del model, fmodel
    torch.cuda.empty_cache()

    return epsilons, robust_accuracy.tolist()


def plot_robustness_curves(results_dict: dict[str, tuple[list[float], list[float]]], output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex=True, sharey=True)
    fig.tight_layout()
    fig.suptitle(r"Adversarial robustness: accuracy vs adv. noise ($\epsilon$)", fontsize=14, y=1.01)
    plt.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

    for group_name, members in GROUPS.items():
        row, col = GROUP_TO_AX[group_name]
        ax = axs[row, col]

        for label, ckpt_name in members.items():
            data = results_dict.get(ckpt_name)
            if data is None:
                print(f"Warning: Checkpoint {ckpt_name} not found in results_dict. Skipping.")
                continue
            eps, acc = data
            linestyle = LABEL_TO_LINESTYLE[label.removesuffix("Adapt")]
            color = LABEL_TO_COLOR[label]
            marker = "v" if "Adapt" in label else "."
            ax.plot(eps, acc, marker=marker, label=label, linestyle=linestyle, color=color)

        if row:
            ax.set_xlabel(r"Perturbation magnitude ($\epsilon$)")
        if not col:
            ax.set_ylabel("Accuracy")
        ax.grid(visible=True, linestyle="--")
        ax.legend(loc="upper right", title=group_name.replace("_", " "), fontsize=8 if group_name == "BMC_AdamW" else 10)

    fig.subplots_adjust(wspace=None, hspace=None)
    plot_path = output_dir / "robustness_curve.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot dynamically updated: {plot_path}")


def main(
    target_count: int | None = 1024,
    test_batch_size: int = 16,
    output_dir: str = "plots/robustness",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print("Starting Robustness Evaluation Script...")
    out_path = PROJECT_ROOT / output_dir

    # checkpoints = get_checkpoints_to_plot()

    all_results = dict[str, tuple[list[float], list[float]]]()

    json_path = out_path / "robustness_results.json"
    if json_path.exists():
        all_results = load_json(json_path)
        print(f"Loaded existing partial results for: {list(all_results.keys())}")

    # for name, ckpt_path in checkpoints.items():
    #     if name in all_results:
    #         print(f"Skipping {name}, already evaluated.")
    #         continue

    #     print(f"\nEntering into {name}...")

    #     eps, acc = run_adversarial_eval(
    #         checkpoint_path=ckpt_path,
    #         target_count=target_count,
    #         test_batch_size=test_batch_size,
    #         device=device,
    #     )
    #     all_results[name] = (eps, acc)

    #     plot_robustness_curves(all_results, out_path)
    #     save_json_atomic(all_results, json_path)

    plot_robustness_curves(all_results, out_path)
    print("\nAll models evaluated successfully!")


if __name__ == "__main__":
    CLI().before_main()
    tyro.cli(main)
