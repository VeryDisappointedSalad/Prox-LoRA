import torch
from torch.optim import Optimizer
import torch.nn.functional as F

from timm.optim._optim_factory import default_registry
from timm.optim._optim_factory import OptimInfo

class ProxSAM(Optimizer):
    """    
    Coding:
    to use in config: OptimizerConfig('opt' = 'prox_sam', lr=1e-3, rho=0.05, prox_lambda=0.01)

    Math:
    1. Perturbation (First Ascent): 
       theta_adv = theta + epsilon, where epsilon = rho * grad / ||grad||
    2. Sharpness Gradient:
       g_sam = grad(Loss(theta_adv))
    3. Descent & Proximal Step (Backward):
       theta_{t+1} = prox_{lr * lambda}(theta - lr * g_sam)
    """

    def __init__(self, params, lr=1e-3, rho=0.05, prox_lambda=0.01, eps=1e-12, **kwargs):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if rho < 0:
            raise ValueError(f"Invalid rho (perturbation radius): {rho}")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda}")

        defaults = dict(lr=lr, rho=rho, prox_lambda=prox_lambda, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure):

        if closure is None:
            raise ValueError("Prox-SAM requires a closure to calculate gradients at perturbed points.")

        loss = None
        
        for group in self.param_groups:
            rho = group["rho"]
            eps = group["eps"]
            lr = group["lr"]
            prox_lambda = group["prox_lambda"]

            # epsilon = rho * grad / ||grad||_2
            for p in group["params"]:
                if p.grad is None: continue
                
                grad_norm = p.grad.norm(p=2)
                scale = rho / (grad_norm + eps)
                
                # epsilon in state
                state = self.state[p]
                state["e"] = p.grad * scale
                
                p.add_(state["e"])

            # grad at perturbed point g_sam = grad(f(theta + epsilon))
            with torch.enable_grad():
                loss = closure()

            # apply descent + prox
            for p in group["params"]:
                if p.grad is None: continue
                
                state = self.state[p]
                
                # restore original weights before taking the step
                # theta = theta_adv - epsilon
                p.sub_(state["e"])
                
                # gradient step using the SAM gradient
                # theta' = theta - lr * g_sam
                p.add_(p.grad, alpha=-lr)
                
                # proximal step L1 soft-thresholding
                # theta_{t+1} = sgn(theta') * max(|theta'| - lr * lambda, 0)
                if prox_lambda > 0:
                    threshold = lr * prox_lambda
                    p.copy_(F.softshrink(p, threshold))

        return loss

# Register Prox-SAM optimizer
info = OptimInfo(name="ProxSam", opt_class=ProxSAM, description="Sharpness-Aware Proximal Optimizer")
default_registry.register(info)