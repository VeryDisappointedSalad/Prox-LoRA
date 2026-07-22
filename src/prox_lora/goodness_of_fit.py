import json
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


def run_clean_eval(
    checkpoint_path: str, config: FullTrainConfig, test_batch_size: int, device: str = "cuda"
) -> dict[str, float]:

    actual_device = device
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: GPU not available! Using CPU.")
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

    num_classes = config.model.num_classes
    acc_metric = MulticlassAccuracy(num_classes=num_classes).to(actual_device)
    kappa_metric = MulticlassCohenKappa(num_classes=num_classes, weights="quadratic").to(actual_device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average="macro").to(actual_device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average="macro").to(actual_device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average="macro").to(actual_device)

    print(f"Running Clean Evaluation on: {checkpoint_path}")

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(actual_device), labels.to(actual_device)

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

    del model, model_module
    torch.cuda.empty_cache()

    return metrics_results


def save_evaluation_results(results_dict: dict[str, dict[str, float]], output_dir: Path) -> None:
    df = pd.DataFrame.from_dict(results_dict, orient="index")
    csv_path = output_dir / "evaluation_metrics.csv"
    df.to_csv(csv_path, index_label="Model")

    json_path = output_dir / "evaluation_metrics.json"
    with open(json_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print("\nCurrent Results Table:")
    print(df.to_string())


def main(
    test_batch_size: int = 32,
    output_dir: str = "plots/robustness",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> None:

    print("Starting Standard Model Evaluation Script...")
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
            # I really like this emoji
            print(f"🔎 Found latest checkpoint for {name}: {latest_ckpt.relative_to(PROJECT_ROOT)}")
        else:
            print(f"No checkpoint found in: {dir_path.relative_to(PROJECT_ROOT) if dir_path.exists() else dir_path}")

    all_results = {}

    json_path = out_path / "evaluation_metrics.json"
    if json_path.exists():
        try:
            with open(json_path) as f:
                all_results = json.load(f)
                print(f"Loaded existing partial results for: {list(all_results.keys())}")
        except Exception:
            print("Could not read previous JSON, starting clean evaluation from scratch.")

    for name, full_ckpt_path in checkpoints.items():
        if name in all_results:
            print(f"Skipping {name}, already evaluated.")
            continue

        config_path = full_ckpt_path.parent.parent / "config.yaml"

        if full_ckpt_path.exists() and config_path.exists():
            print(f"\nEntering into {name}...")
            run_config = load_config(config_path)

            model_metrics = run_clean_eval(
                checkpoint_path=str(full_ckpt_path), config=run_config, test_batch_size=test_batch_size, device=device
            )
            all_results[name] = model_metrics

            save_evaluation_results(all_results, out_path)
        else:
            print(f"Skipping {name}, checkpoint or config.yaml not found at: {config_path}")

    print(f"\nFinal evaluation completed! Reports saved to {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
