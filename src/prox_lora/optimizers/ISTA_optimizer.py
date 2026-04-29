from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from timm.optim._optim_factory import OptimInfo, default_registry
from torch import nn
from torch.optim import Optimizer


class ISTA(Optimizer):
    """
    Usage: `OptimizerConfig(opt="ista", ...)`.

    Math:
        solves min_θ ℒ(θ) + λ · L1_regularization(θ)

        θ' = θₜ - lr · ∇_θₜ ℒ(θₜ)
        θₜ₊₁ = sgn(θ') · max(|θ'| - prox_lambda · lr, 0)
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        momentum: float = 0,
        weight_decay: float = 0,
        prox_lambda: float = 0.01,
        **kwargs: Any,
    ) -> None:

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr} < 0")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda} < 0")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, prox_lambda=prox_lambda)
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        # print("ISTA OPTIMIZER IS RUNNING!")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            prox_lambda = group["prox_lambda"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # get the grad
                d_p = p.grad

                # do weight decay if specified, apparently PyTorch normally does this
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # gradient step
                # p' = p - lr * grad
                p.add_(d_p, alpha=-lr)

                # proximal step, soft thresholding for L1
                # prox(w) = sign(w) * max(|w| - lambda * lr, 0)
                if prox_lambda > 0:
                    threshold = prox_lambda * lr
                    p.copy_(torch.sign(p) * torch.maximum(p.abs() - threshold, torch.tensor(0.0, device=p.device)))

        return loss


# Register ISTA optimizer
info = OptimInfo(name="ista", opt_class=ISTA, description="Custom ISTA Optimizer")
default_registry.register(info)
