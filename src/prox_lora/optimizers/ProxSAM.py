from collections.abc import Callable
from typing import Any, cast, overload

import torch
import torch.nn.functional as F
from timm.optim._optim_factory import OptimInfo, default_registry
from timm.optim.sgdw import SGDW
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from prox_lora.optimizers.common import FloatScalar


class ProxSAM(Optimizer):
    """
    Coding:
    To use in config: OptimizerConfig('opt' = 'proxsam', lr=1e-3, weight_decay=1e-4, momentum=0.9, rho=0.05, prox_lambda=0.01)

    Math:
    1. Perturbation (First Ascent):
       w_{t, adv} = w_t + \\hat{\\epsilon}_t, where \\hat{\\epsilon}_t = \rho * \nabla L(w_t) / ||\nabla L(w_t)||_2
    2. Sharpness Gradient:
       g^{SAM}_t = \nabla L(w_{t, adv})
    3. Interim Descent (Forward):
       w'_{t+1} = w_t - \\eta * g^{SAM}_t  (with optional momentum)
    4. Proximal Step (Backward):
       w_{t+1} = prox_{\\eta R}(w'_{t+1})  (handles L1 and/or L2)
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        rho: float = 0.05,
        prox_lambda: float = 0.01,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        eps: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho < 0:
            raise ValueError(f"Invalid rho (perturbation radius): {rho}")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda}")

        defaults = dict(lr=lr, rho=rho, prox_lambda=prox_lambda, weight_decay=weight_decay, momentum=momentum, eps=eps)
        super().__init__(params, defaults)

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], FloatScalar]) -> FloatScalar: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], FloatScalar] | None = None) -> FloatScalar | None:
        if closure is None:
            raise ValueError("Prox-SAM requires a closure to calculate gradients at perturbed points.")
        with torch.enable_grad():
            loss = closure()

        # SAM
        grad_norm = self._grad_norm()
        any_rho = any(group["rho"] > 0 for group in self.param_groups)
        if any_rho:
            for group in self.param_groups:
                rho = group["rho"]
                eps = group["eps"]

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    self.state[p]["old_p"] = p.data.clone()

                    # Move to adversarial point:
                    # p += \hat{ε}_t
                    # where \hat{ε}_t = p.grad * scale
                    p.add_(p.grad, alpha=rho / (grad_norm + eps))

            # Grad evaluation at perturbed point.
            with torch.enable_grad():
                loss = closure()

        # Descent and proximal step.
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            prox_lambda = group["prox_lambda"]
            rho = group["rho"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # Restore original weights from time step t, after SAM step.
                if rho:
                    p.copy_(state["old_p"])

                g_sam = p.grad

                if momentum > 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.clone(g_sam).detach()
                    else:
                        state["momentum_buffer"].mul_(momentum).add_(g_sam)
                    g_sam = state["momentum_buffer"]

                # Proximal L2 (Decoupled Weight Decay / Ridge)
                if weight_decay > 0:
                    # w_{t+1} = w'_{t+1} / (1 + \eta * \lambda)
                    p.mul_(1.0 - lr * weight_decay)

                # Descent step: w'_{t+1} = w_t - \eta * g^{SAM}_t
                p.add_(g_sam, alpha=-lr)

                # Proximal L1 (Soft-thresholding / Lasso / Sparsity)
                if prox_lambda > 0:
                    # w_{t+1} = sgn(w'_{t+1}) * max(|w'_{t+1}| - \eta * \lambda_1, 0)
                    threshold = prox_lambda * lr
                    p.copy_(F.softshrink(p, threshold))

        return loss

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return cast(torch.Tensor, torch.linalg.vector_norm(torch.stack(norms), ord=2))


# Register Prox-SAM optimizer
info = OptimInfo(name="proxsam", opt_class=ProxSAM, has_momentum=True, description="Sharpness-Aware Proximal Optimizer")
default_registry.register(info)

default_registry.register(
    OptimInfo(
        name="sgdw-nonesterov",
        opt_class=SGDW,
        description="SGD with decoupled weight decay and non-Nesterov momentum",
        has_eps=False,
        has_momentum=True,
        defaults={"nesterov": False},
    )
)
