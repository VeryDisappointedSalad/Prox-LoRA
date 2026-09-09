from pathlib import Path

import pandas as pd
import torch
import tyro
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassCohenKappa,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from tqdm import tqdm

from prox_lora.infrastructure.cli import CLI
from prox_lora.utils.eval import get_checkpoints_to_plot, load_for_eval
from prox_lora.utils.io import PROJECT_ROOT, load_json, save_json_atomic


def run_clean_eval(
    checkpoint_path: Path, test_batch_size: int, device: str = "cuda"
) -> dict[str, float]:
    config, model, test_loader = load_for_eval(Path(checkpoint_path), test_batch_size, device)

    num_classes = config.model.num_classes
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average="macro").to(device)
    kappa_metric = MulticlassCohenKappa(num_classes=num_classes, weights="quadratic").to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro").to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(device)

    print(f"Running Clean Evaluation on: {checkpoint_path}")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=-1)

            acc_metric.update(preds, labels)
            kappa_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)

    metrics_results = {
        "Accuracy": round(float(acc_metric.compute().item()), 4),
        "Quadratic_Kappa": round(float(kappa_metric.compute().item()), 4),
        "F1_Score_Macro": round(float(f1_metric.compute().item()), 4),
        "Precision_Macro": round(float(precision_metric.compute().item()), 4),
        "Recall_Macro": round(float(recall_metric.compute().item()), 4),
    }

    del model
    torch.cuda.empty_cache()

    return metrics_results




def main(
    test_batch_size: int = 32,
    output_dir: str = "plots/robustness",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:
    print("Starting Standard Model Evaluation Script...")
    out_path = PROJECT_ROOT / output_dir
    out_path.mkdir(exist_ok=True, parents=True)

    checkpoints = get_checkpoints_to_plot()

    all_results = {}

    json_path = out_path / "evaluation_metrics.json"
    if json_path.exists():
        all_results = load_json(json_path)
        print(f"Loaded existing partial results for: {list(all_results.keys())}")

    for name, ckpt_path in checkpoints.items():
        if name in all_results:
            print(f"Skipping {name}, already evaluated.")
            continue

        print(f"\nEntering into {name}...")

        model_metrics = run_clean_eval(
            checkpoint_path=ckpt_path, test_batch_size=test_batch_size, device=device
        )
        all_results[name] = model_metrics

        save_json_atomic(all_results, json_path)

        df = pd.DataFrame.from_dict(all_results, orient="index")
        df.to_csv(json_path.with_suffix(".csv"), index_label="Model")
        print("\nCurrent Results Table:")
        print(df.to_string())

    print(f"\nFinal evaluation completed! Reports saved to {out_path}")


if __name__ == "__main__":
    CLI().before_main()
    tyro.cli(main)
