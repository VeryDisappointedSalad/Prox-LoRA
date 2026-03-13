from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from timm.optim._optim_factory import OptimInfo, default_registry
from torch import nn
from torch.optim import Optimizer


# @default_registry.register
class FISTA(Optimizer):
    """
    Coding:
    to use in config: OptimizerConfig(opt="fista", ...)

    Math:
    solves min_theta Loss_function(theta) + lambda * L1_regularization(theta)

    y_{t+1} = prox(x_t - lr * grad)
    y_{t+1} = sgn(x_t) * max(|abs(x_t)|-prox_lambda * lr, 0)
    x_{t+1} = y_{t+1} + ((t-1)/(t+2)) * (y_{t+1} - y_t)
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
        # print("FISTA OPTIMIZER IS RUNNING!")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            prox_lambda = group["prox_lambda"]
            lr = group["lr"]
            _momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # get the grad
                d_p = p.grad

                # initialize counter and state
                state = self.state[p]
                if len(state) == 0:
                    state["t"] = 1
                    state["y_prev"] = p.clone().detach()  # the y_t

                # get current state
                t = state["t"]
                y_prev = state["y_prev"]

                # do weight decay if specified, apparently PyTorch normally does this
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # gradient step from x_t
                # p' = p - lr * grad
                z = p - lr * d_p

                # proximal step, soft thresholding for L1
                # prox(w) = sign(w) * max(|w| - lambda * lr, 0)
                if prox_lambda > 0:
                    threshold = prox_lambda * lr
                    # y_{t+1}
                    y_next = torch.sign(z) * torch.maximum(z.abs() - threshold, torch.tensor(0.0, device=p.device))
                else:
                    y_next = z

                # Nesterov like momentum, the t-1/t+1 part
                nesterov_momentum = (t - 1) / (t + 2)

                # x_{t+1}
                x_next = y_next + nesterov_momentum * (y_next - y_prev)

                # store for next iteration
                p.copy_(x_next)
                state["y_prev"] = y_next
                state["t"] += 1

        return loss


info = OptimInfo(name="fista", opt_class=FISTA, description="Custom FISTA Optimizer")
default_registry.register(info)
