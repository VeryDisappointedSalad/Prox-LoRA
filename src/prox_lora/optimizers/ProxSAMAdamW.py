import torch
import torch.nn.functional as F
from timm.optim._optim_factory import OptimInfo, default_registry
from torch.optim import Optimizer


class ProxSAMAdamW(Optimizer):
    """
    ProxSAM-AdamW Optimizer: Combines Sharpness-Aware Minimization (SAM),
    AdamW adaptive learning rates, and generalized proximal operators (L1 & L2).

    To use in config:
    OptimizerConfig('opt' = 'proxsam_adamw', lr=1e-3, beta1=0.9, beta2=0.999, weight_decay=1e-4, rho=0.01, prox_lambda=1e-5)
    """

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
    def step(self, closure):
        if closure is None:
            raise ValueError("ProxSAM-AdamW requires a closure to calculate gradients at perturbed points.")

        grad_norm = self._grad_norm()

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

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            prox_lambda = group["prox_lambda"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                p.sub_(state["eps_hat"])

                g_sam = p.grad

                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g_sam, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_sam, g_sam, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                denom = (exp_avg_sq.sqrt() / (bias_correction2**0.5)).add_(eps)
                step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

                if weight_decay > 0:
                    p.div_(1.0 + lr * weight_decay)

                if prox_lambda > 0:
                    threshold = lr * prox_lambda
                    p.copy_(F.softshrink(p, threshold))

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


info = OptimInfo(name="proxsamadw", opt_class=ProxSAMAdamW, description="Sharpness-Aware Proximal AdamW Optimizer")
default_registry.register(info)
