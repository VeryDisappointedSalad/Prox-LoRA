import torch
import foolbox as fb
import matplotlib.pyplot as plt
from pathlib import Path

from prox_lora.models.classifier import Classifier
from prox_lora.datasets.diabetic_retinopathy import DRDataModule
from prox_lora.infrastructure.configs import get_config
from prox_lora.infrastructure.trainer import FullTrainConfig
from prox_lora.utils.io import PROJECT_ROOT


def run_adversarial_eval(checkpoint_path: str, config: FullTrainConfig, device="cuda"):

    # setup the model
    model_instance = config.model.instantiate()
    optimizer_cfg = config.optimizer
    scheduler_cfg = config.scheduler
    
    model_module = Classifier.load_from_checkpoint(
        checkpoint_path,
        model=model_instance,
        optimizer=optimizer_cfg,
        scheduler=scheduler_cfg
    )
    model = model_module.model.to(device).eval()
    
    # get data
    datamodule = DRDataModule(dataloader=config.dataloader, augmentations=False)
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    # attacks
    mean = [0.4814, 0.4578, 0.4082]
    std = [0.2686, 0.2613, 0.2757]
    preprocessing = dict(mean=mean, std=std, axis=-3)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device, preprocessing=preprocessing)
    attack = fb.attacks.LinfPGD()
    epsilons = [0.0, 0.0001, 0.001, 0.01, 0.02]

    # batching the overall attacks, if not performed with DRDataModule.batch_size it takes forever
    total_success = []
    target_count = 64
    current_count = 0

    print(f"PGD Attack on: {checkpoint_path}")
    
    mean_t = torch.tensor(mean).view(3, 1, 1).to(device)
    std_t = torch.tensor(std).view(3, 1, 1).to(device)

    for batch_idx, (images, labels) in enumerate(test_loader):

        images, labels = images.to(device), labels.to(device)
        
        # umnormalize for the attack
        unnormalized = (images * std_t + mean_t).clamp(0, 1)

        # attack on this batch
        _, _, success = attack(fmodel, unnormalized, labels, epsilons=epsilons)
        
        total_success.append(success.cpu()) # gpu RAM is precious
        current_count += images.size(0)
        print(f"Batch {batch_idx+1}: Processed {current_count}/{target_count} images...")
        
        if current_count >= target_count:
            print(f"\n Evaluated {current_count} images.")
            break

    # combine all results
    combined_success = torch.cat(total_success, dim=-1)[:, :target_count]
    robust_accuracy = 1.0 - combined_success.float().mean(axis=-1).numpy()

    # clean GPU memory for the next model in the loop
    del model, fmodel
    torch.cuda.empty_cache()

    return epsilons, robust_accuracy

def plot_robustness_curves(results_dict: dict): # TODO: move to something like utils.plots ? # TODO 2 : be able to plot partial results
    
    Path("Plots").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for model_name, (eps, acc) in results_dict.items():
        plt.plot(eps, acc, marker='o', label=model_name)
    
    plt.title(r"Robustness Curve: Accuracy vs. Adversarial Noise ($\epsilon$)")
    plt.xlabel(r"Perturbation Magnitude ($\epsilon$)")
    plt.ylabel("Accuracy")

    plt.grid(True, linestyle='--')
    plt.legend()

    plt.savefig("Plots/robustness_curve.png")
    print("Plot saved to Plots/robustness_curve.png")

if __name__ == "__main__":

    print('Starting Robustness Evaluation Script...')

    print('CNN config.')
    cnn_config = get_config(FullTrainConfig, "CNN_retinopathy")

    print('BioMed config.')
    biomed_config = get_config(FullTrainConfig, "bmc_example")

    print('Got configs.')

    # TODO: Right now it's by hand, like commenting out the part of dict you don't want xd
    # add it to some config or make accessible in run_eval.sh, especially int : target_count which defines number of iterations to be run

    # TODO 2: save even partial results, biomedclip takes a lot of time for inference

    checkpoints = {
        # --- BioMedCLIP Models (Transformer) ---
        "BioMed Adam": ("checkpoints/BioMed_DR_adam/version_0/checkpoints/last.ckpt", biomed_config),
        "BioMed ADMM": ("checkpoints/BioMed_DR_admm/version_0/checkpoints/last.ckpt", biomed_config),
        "BioMed FISTA": ("checkpoints/BioMed_DR_FISTA/version_0/checkpoints/last.ckpt", biomed_config),
        "BioMed ISTA": ("checkpoints/BioMed_DR_ista/version_0/checkpoints/last.ckpt", biomed_config),
        "BioMed ProxAdam": ("checkpoints/BioMed_DR_proxAdam/version_0/checkpoints/last.ckpt", biomed_config),
        "BioMed SGD": ("checkpoints/BioMed_DR_sgd/version_0/checkpoints/last.ckpt", biomed_config),

        # --- CNN Models (ExampleCNN) ---
        #"CNN AdaProx": ("checkpoints/CNN_DR_adaprox/version_0/checkpoints/last.ckpt", cnn_config),
        #"CNN FISTA": ("checkpoints/CNN_DR_FISTA/version_0/checkpoints/last.ckpt", cnn_config),
        #"CNN ISTA": ("checkpoints/CNN_DR_ISTA/version_0/checkpoints/last.ckpt", cnn_config),
        #"CNN ProxAdam": ("checkpoints/CNN_DR_proxAdam/version_0/checkpoints/last.ckpt", cnn_config),
        #"CNN SGD": ("checkpoints/CNN_DR_sgd/version_0/checkpoints/last.ckpt", cnn_config),
        #"CNN ProxSam": ("checkpoints/CNN_DR_ProxSam/version_0/checkpoints/last.ckpt", cnn_config),
    }
    
    all_results = {}
    for name, (path, config) in checkpoints.items():
        if Path(path).exists():
            print(f'Entering into {path}...')
            eps, acc = run_adversarial_eval(path, config)
            all_results[name] = (eps, acc)
        else:
            print(f"Skipping {name}, checkpoint not found at {path}")
            
    plot_robustness_curves(all_results)