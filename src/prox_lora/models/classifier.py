from __future__ import annotations

from copy import replace
from dataclasses import asdict, dataclass
from typing import Literal, cast

import timm.scheduler.scheduler
import torch.nn as nn
import torch.optim
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerConfig as LightningOptimizerConfig
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from torch import Tensor
from torchmetrics.classification import CohenKappa

from prox_lora.infrastructure.configs import yaml
from prox_lora.optimizers.common import OptimizerConfig, SchedulerConfig


class Classifier(LightningModule):
    """
    LightningModule for training a classifier.

    - model: am nn.Module that takes inputs of shape (B, C, H, W) and produces logits of shape (B, num_classes).
    - optimizer_kwargs: passed to timm.optim.create_optimizer_v2().
        Common/default args are opt="sgd" (or e.g. "adamw"), lr, weight_decay=0, momentum=0.9.
        See: https://huggingface.co/docs/timm/reference/optimizers#timm.optim.create_optimizer_v2
        Note that sgd means SGD with Nesterov momentum in timm (use opt="momentum" to disable Nesterov).
    - scheduler_kwargs: passed to timm.scheduler.create_scheduler_v2()
        Common/default args are sched="cosine", num_epochs=300, decay_epochs=90,
            decay_milestones=[90, 180, 270], cooldown_epoch=0, patience_epochs=10, decay_rate=0.1,
            min_lr=0, warmup_lr=1e-05, warmup_epochs=0.
        See: https://huggingface.co/docs/timm/reference/schedulers#timm.scheduler.create_scheduler_v2
    - steps_in_epoch: Number of batches in an epoch, used to convert scheduler args from epochs to steps if needed.
    - loss_ce_alpha: multiplier for CE loss.
    - loss_class_weights_gamma: if non-zero, use class weights in the cross-entropy loss to counter class imbalance.
        Weights are computed as freq^gamma for each class.
    - class_frequencies: List of frequencies for each class, only used with `loss_class_weights`.
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        optimizer: OptimizerConfig,
        scheduler: SchedulerConfig,
        steps_in_epoch: int = 0,
        *,
        loss_ce_alpha: float = 1.0,
        loss_class_weights_gamma: float = 0.0,
        continuous_kappa: ContinuousKappaConfig | None = None,
        class_frequencies: list[float],
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer
        if not scheduler.step_on_epochs and not scheduler.updates_per_epoch and steps_in_epoch:
            scheduler = replace(scheduler, updates_per_epoch=steps_in_epoch)
        self.scheduler_config = scheduler

        self.steps_in_epoch = steps_in_epoch

        # Disable training_step wrapper that implicitly adds: zero_grad, backward, optimizer and scheduler step, gradient clipping, ...
        self.automatic_optimization = False

        self.train_kappa = CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic")
        self.val_kappa = CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic")
        self.test_kappa = CohenKappa(task="multiclass", num_classes=num_classes, weights="quadratic")

        self.class_frequencies = class_frequencies
        self.class_weights: Tensor | None
        if loss_class_weights_gamma != 0.0:
            class_weights = [freq**loss_class_weights_gamma for freq in class_frequencies]
            # Normalize to sum to num_classes.
            class_weights = [num_classes * w / sum(class_weights) for w in class_weights]
            self.register_buffer("class_weights", torch.tensor(class_weights))
        else:
            self.class_weights = None

        self.loss_ce_alpha = loss_ce_alpha
        self.continuous_kappa_config = continuous_kappa

    def compute_loss(self, batch: tuple[Tensor, Tensor], phase: Literal["train", "val", "test"]) -> Tensor:
        inputs, targets = batch
        batch_size = len(inputs)

        logits = self.model(inputs)
        ce_loss = nn.functional.cross_entropy(logits, targets, weight=self.class_weights)
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()
        loss = ce_loss

        if self.continuous_kappa_config:
            ckappa_loss = 1.0 - continuous_kappa(
                logits,
                targets,
                temperature=self.continuous_kappa_config.temperature,
                mu=self.continuous_kappa_config.mu,
                class_frequencies=self.class_frequencies,
                class_weights=self.class_weights if self.continuous_kappa_config.use_class_weights else None,
            )
            self.log(f"_loss/ckappa/{phase}", ckappa_loss, batch_size=batch_size)

            loss = self.loss_ce_alpha * loss + self.continuous_kappa_config.alpha * ckappa_loss

        self.log(f"_loss/ce/{phase}", ce_loss, batch_size=batch_size)
        self.log(f"_loss/{phase}", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"_accuracy/{phase}", accuracy, prog_bar=True, batch_size=batch_size)

        if phase == "train":
            kappa = self.train_kappa(predictions, targets)
            self.log(f"_kappa/{phase}", kappa, prog_bar=True, batch_size=batch_size)
        elif phase == "val":
            self.val_kappa.update(predictions, targets)
        elif phase == "test":
            self.test_kappa.update(predictions, targets)

        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        optimizer = cast(torch.optim.Optimizer, self.optimizers())

        optimizer.zero_grad()

        loss = self.compute_loss(batch, phase="train")

        self.manual_backward(loss)

        opt_name = (
            self.optimizer_config["opt"] if isinstance(self.optimizer_config, dict) else self.optimizer_config.opt
        )
        if opt_name in ["proxsam", "proxsamadw", "proxsamadaptive"]:

            def sam_closure() -> Tensor:
                optimizer.zero_grad()
                # with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                inputs, targets = batch
                logits = self.model(inputs)
                adv_loss = nn.functional.cross_entropy(logits, targets, weight=self.class_weights)
                adv_loss.backward()  # type: ignore[no-untyped-call]
                return adv_loss

            optimizer.step(closure=sam_closure)  # type: ignore[arg-type] # mistyped return type of callback.
        else:
            optimizer.step()

        self._step_scheduler("batch")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self.compute_loss(batch, phase="val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self.compute_loss(batch, phase="test")

    def on_train_epoch_end(self) -> None:
        self._step_scheduler("epoch")
        self.train_kappa.reset()

    def on_validation_epoch_end(self) -> None:
        self.log("_kappa/val", cast(Tensor, self.val_kappa.compute()), prog_bar=True)
        self.val_kappa.reset()

    def on_test_epoch_end(self) -> None:
        self.log("_kappa/test", cast(Tensor, self.test_kappa.compute()), prog_bar=True)
        self.test_kappa.reset()

    def _step_scheduler(self, kind: Literal["batch", "epoch"]) -> None:
        lr_scheduler = cast(timm.scheduler.scheduler.Scheduler, self.lr_schedulers())
        if lr_scheduler is None:
            return
        try:
            metric = self.trainer.callback_metrics["_loss/val"].item()  # Metric used for PlateauLRScheduler.
        except KeyError:
            metric = 0.0
        if kind == "epoch":
            lr_scheduler.step(epoch=self.current_epoch, metric=metric)
        else:
            lr_scheduler.step_update(num_updates=self.global_step, metric=metric)

    def configure_optimizers(self) -> LightningOptimizerConfig | OptimizerLRSchedulerConfig:
        # If we wanted, for example, different lr for head vs. backbone, we could do:
        #     lr = optimizer_kwargs.pop("lr")
        #     lr_head = optimizer_kwargs.pop("lr_head")
        #     create_optimizer_v2([
        #         {"params": self.model.backbone.parameters(), "lr": lr},
        #         {"params": self.model.head.parameters(), "lr": lr_head}
        #     ], **optimizer_kwargs)

        optimizer = create_optimizer_v2(self.model, **self.optimizer_config)
        if self.scheduler_config.sched == "none":
            return {"optimizer": optimizer}
        else:
            scheduler, _num_epochs = create_scheduler_v2(optimizer, **asdict(self.scheduler_config))
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch" if self.scheduler_config.step_on_epochs else "step",
                },
            }


@yaml.register_class
@dataclass(frozen=True)
class ContinuousKappaConfig:
    alpha: float = 1.0
    """Weighting factor for the loss, in the total loss."""

    temperature: float = 1.0
    """Temperature for the softmax function."""

    mu: float = 0.5
    """
    How much of the global dataset frequencies to mix in when computing the expected confusion matrix.
    0 means only use the batch frequencies (which can be unstable/high variance),
    1 means only use the global frequencies, effectively computing MSE loss.
    """

    use_class_weights: bool = False
    """Whether to use class weights (see loss_class_weights_gamma) for this loss."""


def continuous_kappa(
    logits: Tensor,
    targets: Tensor,
    *,
    temperature: float = 1.0,
    mu: float = 0.5,
    class_frequencies: list[float],
    class_weights: Tensor | None = None,
) -> Tensor:
    """
    Args:
        logits: (B, num_classes) float tensor of model outputs, before softmax.
        targets: (B,) int tensor of ground truth labels.

        class_frequencies: List of true class frequencies in the whole dataset.
    """
    B, num_classes = logits.shape
    probs = nn.functional.softmax(logits / temperature, dim=-1)

    # Compute the confusion matrix: obs[i, j] = joint probability that an item was true class i and predicted as class j.
    obs = torch.zeros((num_classes, num_classes), device=logits.device)
    for i in range(B):
        obs[targets[i], :] += probs[i, :] / B

    # Compute the expected confusion matrix:
    pred_dist = probs.mean(dim=0)  # (num_classes,)
    true_dist = torch.zeros(num_classes, device=logits.device)  # true distribution in batch.
    class_dist = torch.tensor(class_frequencies, device=logits.device)  # true distribution in whole dataset.
    for i in range(B):
        true_dist[targets[i]] += 1.0 / B
    pred_dist = (1 - mu) * pred_dist + mu * class_dist
    true_dist = (1 - mu) * true_dist + mu * class_dist
    exp = torch.outer(true_dist, pred_dist)  # (num_classes, num_classes)

    # Compute the quadratic cost matrix for Cohen's kappa.
    cost = quadratic_cost_matrix(num_classes, class_weights).to(logits.device)

    print((cost * exp).sum())

    return 1.0 - (cost * obs).sum() / (cost * exp).sum()


def quadratic_cost_matrix(num_classes: int, class_weights: Tensor | None) -> Tensor:
    """Compute the quadratic cost matrix for Cohen's kappa."""
    if class_weights is not None:
        return torch.tensor(
            [
                [(class_weights[i] * (i - j) ** 2) / ((num_classes - 1) ** 2) for j in range(num_classes)]
                for i in range(num_classes)
            ]
        )
    else:
        return torch.tensor(
            [[((i - j) ** 2) / ((num_classes - 1) ** 2) for j in range(num_classes)] for i in range(num_classes)]
        )
