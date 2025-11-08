from dataclasses import asdict
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
    """

    def __init__(self, model: nn.Module, optimizer: OptimizerConfig, scheduler: SchedulerConfig) -> None:
        super().__init__()
        self.model = model
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Disable training_step wrapper that implicitly adds: zero_grad, backward, optimizer and scheduler step, gradient clipping, ...
        self.automatic_optimization = False

    def compute_loss(self, batch: tuple[Tensor, Tensor], phase: Literal["train", "val", "test"]) -> Tensor:
        inputs, targets = batch
        batch_size = len(inputs)

        logits = self.model(inputs)
        loss = nn.functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()

        self.log(f"_loss/{phase}", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"_accuracy/{phase}", accuracy, prog_bar=True, batch_size=batch_size)

        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        optimizer = cast(torch.optim.Optimizer, self.optimizers())

        optimizer.zero_grad()

        loss = self.compute_loss(batch, phase="train")

        self.manual_backward(loss)

        optimizer.step()

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self.compute_loss(batch, phase="val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        self.compute_loss(batch, phase="test")

    def on_train_epoch_end(self) -> None:
        lr_scheduler = cast(timm.scheduler.scheduler.Scheduler, self.lr_schedulers())
        metric = self.trainer.callback_metrics["_loss/val"]  # Metric used for PlateauLRScheduler.
        lr_scheduler.step(epoch=self.current_epoch, metric=metric.item())

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
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler}}
