import torch
import torch.nn.functional as F
from timm.optim._optim_factory import OptimInfo, default_registry
from torch.optim import Optimizer


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
        self, params, lr=1e-3, rho=0.05, prox_lambda=0.01, weight_decay=0.0, momentum=0.0, eps=1e-12, **kwargs
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho < 0:
            raise ValueError(f"Invalid rho (perturbation radius): {rho}")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda}")

        defaults = dict(lr=lr, rho=rho, prox_lambda=prox_lambda, weight_decay=weight_decay, momentum=momentum, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):
        if closure is None:
            raise ValueError("Prox-SAM requires a closure to calculate gradients at perturbed points.")

        grad_norm = self._grad_norm()

        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]

            # \hat{\epsilon}_t scale factor
            scale = rho / (grad_norm + eps)

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # \hat{\epsilon}_t
                state["eps_hat"] = p.grad * scale

                # move to adversarial point
                p.add_(state["eps_hat"])

        with torch.enable_grad():
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            prox_lambda = group["prox_lambda"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]

                # original weights from time step t ---> w_t = w_{t, adv} - \hat{\epsilon}_t
                p.sub_(state["eps_hat"])

                g_sam = p.grad

                if momentum > 0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.clone(g_sam).detach()
                    else:
                        state["momentum_buffer"].mul_(momentum).add_(g_sam)
                    g_sam = state["momentum_buffer"]

                # Forward: w'_{t+1} = w_t - \eta * g^{SAM}_t
                p.add_(g_sam, alpha=-lr)

                # Backward

                # Proximal L2 (Weight Decay / Ridge)
                if weight_decay > 0:
                    # w_{t+1} = w'_{t+1} / (1 + \eta * \lambda_2)
                    p.div_(1.0 + lr * weight_decay)

                # Proximal L1 (Soft-thresholding / Lasso / Sparsity)
                if prox_lambda > 0:
                    # w_{t+1} = sgn(w'_{t+1}) * max(|w'_{t+1}| - \eta * \lambda_1, 0)
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


# Register Prox-SAM optimizer
info = OptimInfo(name="proxsam", opt_class=ProxSAM, description="Sharpness-Aware Proximal Optimizer")
default_registry.register(info)
