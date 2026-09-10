from pathlib import Path

import torch

from prox_lora.datasets.base_data_module import DataLoaderConfig
from prox_lora.infrastructure.configs import load_config
from prox_lora.infrastructure.trainer import FullTrainConfig
from prox_lora.models.classifier import Classifier
from prox_lora.utils.io import PROJECT_ROOT, find_latest_checkpoint

GROUPS ={
  "BMC_SGD": {
      "base": "bmc4_sgd",
      "ISTA": "bmc4_pl1e-2_rho0",
      "SAM": "bmc4_pl0_rho2e-3_lrx2",
      "ProxSAM": "bmc4_pl1e-2_rho2e-3_lrx2"
  },
  "BMC_AdamW": {
      "base": "bmc4_adamw",
      "ISTA": "bmc4_proxsamadamw_pl1e-2_rho0",
      "SAM": "bmc4_proxsamadamw_pl0_rho2e-3",
      "ProxSAM": "bmc4_proxsamadamw_pl1e-2_rho2e-3",
      "baseAdapt": "bmc4_proxsamadaptive_dummy",
      # "ISTAAdapt": "bmc4_proxsamadaptive_pl1e-2_rho0",
      "SAMAdapt": "bmc4_proxsamadaptive_pl0_rho2e-3",
      # "ProxSAMAdapt": "bmc4_proxsamadaptive_pl1e-2_rho2e-3"
  },
  "conv_SGD": {
      "base": "conv4_sgd_entropy",
      "ISTA": "conv4_pl1e-4_rho0_entropy",
      "SAM": "conv4_pl0_rho2e-2_entropy",
      "ProxSAM": "conv4_pl1e-4_rho2e-2"
  },
  "conv_AdamW": {
      "base": "conv4_proxsamadamw_dummy",
      "ISTA": "conv4_proxsamadamw_pl1e-2_rho0",
      "SAM": "conv4_proxsamadamw_pl0_rho2e-3",
      "ProxSAM": "conv4_proxsamadamw_pl1e-2_rho2e-3",
      "baseAdapt": "conv4_proxsamadaptive_dummy",
      # "ISTAAdapt": "conv4_proxsamadaptive_pl1e-2_rho0",
      "SAMAdapt": "conv4_proxsamadaptive_pl0_rho2e-3",
      # "ProxSAMAdapt": "conv4_proxsamadaptive_pl1e-2_rho2e-3"
      # "base'": "conv4_adamw",
  }
}

def get_checkpoints_to_plot() -> dict[str, Path]:
    runs_root = PROJECT_ROOT / "runs"

    model_directories = {
        # "conv_ProxSAM": runs_root / "conv4_pl1e-4_rho2e-2/r1/"
        # "AdamW_head": runs_root / "biomedclip_dr_AdamW_head_only",
        # # "SGD_head": runs_root / "biomedclip_dr_SGD_head_only",
        # "AdamW_entire": runs_root / "biomedclip_dr_AdamW_entire_model",
        # # "SGD_entire": runs_root / "biomedclip_dr_SGD_entire_model",
        # "ConvNetAdamW": runs_root / "convnet_dr_AdamW",
        # "ProxSamAdaptive_entire": runs_root / "biomedclip_dr_proxsam_adaptive_entire",
        # "ProxSamAdaptive_head": runs_root / "biomedclip_dr_proxsam_adaptive_head",
    }

    for d in sorted(runs_root.iterdir()):
        if d.is_dir() and d.name.startswith(("bmc4_", "conv4_")):
        # if d.is_dir() and d.name.startswith(("bmc4_proxsama","bmc4_adamw")):
        # if d.is_dir() and d.name.startswith(("bmc4_",)) and not d.name.startswith(("bmc4_proxsama","bmc4_adamw")):
        # if d.is_dir() and d.name.startswith(("conv4_",)):
            model_directories[d.name] = d

    checkpoints = {}
    for name, dir_path in model_directories.items():
        latest_ckpt = find_latest_checkpoint(dir_path)
        if latest_ckpt:
            checkpoints[name] = latest_ckpt
            # obj = torch.load(latest_ckpt, map_location="cpu")
            # epoch, step = obj["epoch"], obj["global_step"]
            # assert epoch == 74 if "conv4_" in name else 29, f"Unexpected epoch for {name}: {epoch}"
            print(f"🔎 Found checkpoint for {name}: {latest_ckpt.relative_to(PROJECT_ROOT)}")
        else:
            print(f"No checkpoint found for {name} in: {dir_path}")

    return checkpoints


def load_for_eval(
    checkpoint_path: Path, test_batch_size: int, device: str = "cuda"
) -> tuple[FullTrainConfig, torch.nn.Module, torch.utils.data.DataLoader[tuple[torch.Tensor, int]]]:
    """Load a config, model and test dataloader for evaluation."""
    config = load_config(checkpoint_path.parent.parent / "config.yaml")

    model_instance = config.model.instantiate()
    model_module = Classifier.load_from_checkpoint(
        checkpoint_path,
        model=model_instance,
        num_classes=config.model.num_classes,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
    )
    model = model_module.model.to(device).eval()
    model.compile(backend="inductor")

    eval_dataloader_cfg = DataLoaderConfig(batch_size=test_batch_size, num_workers=4, pin_memory=(device == "cuda"))
    datamodule = config.datamodule.instantiate(dataloader=eval_dataloader_cfg)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    del model_module

    return config, model, test_loader
