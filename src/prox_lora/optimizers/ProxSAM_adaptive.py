import torch
import torch.nn.functional as F
from timm.optim._optim_factory import OptimInfo, default_registry
from torch.optim import Optimizer


class ProxSAMAdaptive(Optimizer):
    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), rho=0.05, prox_lambda=0.01, weight_decay=0.0, eps=1e-8, **kwargs
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if rho < 0:
            raise ValueError(f"Invalid rho (perturbation radius): {rho}")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda}")

        defaults = dict(lr=lr, betas=betas, rho=rho, prox_lambda=prox_lambda, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, sam_closure=None):

        # ProxSamAdaptive needs to overwrite closure very explicitely
        closure = closure if closure is not None else sam_closure
        if closure is None:
            raise ValueError("ProxSAMAdaptive requires a closure to calculate gradients at perturbed points.")

        grad_norm = self._grad_norm()

        # SAM
        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]
            scale = rho / (grad_norm + eps)

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                state["eps_hat"] = p.grad * scale
                p.add_(state["eps_hat"])

        # grad evaluation at perturbed point
        with torch.enable_grad():
            loss = closure()

        # dscent and Proximal Step
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]  # Lambda_2
            prox_lambda = group["prox_lambda"]  # Lambda_1

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # original weights
                p.sub_(state["eps_hat"])

                g_sam = p.grad

                # state initialization
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                # exponential moving averages
                exp_avg.mul_(beta1).add_(g_sam, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_sam, g_sam, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # D_t metric calculation
                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                step_size = lr / bias_correction1

                # Forward Step in D_t space: w'_t = w_t - \eta * D_t^{-1} * m_t
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Adaptive Proximal L2 (Adaptive Weight Decay)
                if weight_decay > 0:
                    # Multiplier: 1 / (1 + (lr * wd) / D_t)
                    adaptive_l2_penalty = (lr * weight_decay) / denom
                    p.div_(1.0 + adaptive_l2_penalty)

                # Adaptive Proximal L1 (Adaptive Sparsity)
                if prox_lambda > 0:
                    # Threshold: (lr * prox_lambda) / D_t
                    adaptive_l1_threshold = (lr * prox_lambda) / denom

                    # w_{t+1} = sign(w) * max(|w| - threshold, 0)
                    p.copy_(torch.sign(p) * F.relu(torch.abs(p) - adaptive_l1_threshold))

        return loss

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norms = [
            p.grad.norm(p=2).to(shared_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)


info = OptimInfo(name="proxsamadaptive", opt_class=ProxSAMAdaptive, description="Preconditioned ProxSAM Optimizer")
default_registry.register(info)
