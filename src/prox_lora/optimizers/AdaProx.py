'''https://arxiv.org/pdf/1910.10094'''

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from timm.optim._optim_factory import default_registry, OptimInfo

class AdaProx(Optimizer):
    """
    AdaProx Optimizer.

    Coding:
    to use in config: OptimizerConfig('opt' = 'adaprox', ...)

    TODO: Math

    1.

    2.

    3.

    4.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, prox_lambda=0.01, 
                 max_sub_iters=5, sub_eps=1e-4, **kwargs):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda}")

        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, prox_lambda=prox_lambda,
                        max_sub_iters=max_sub_iters, sub_eps=sub_eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            prox_lambda = group["prox_lambda"]
            weight_decay = group["weight_decay"]
            max_sub_iters = group["max_sub_iters"]
            sub_eps = group["sub_eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # state initialization
                if len(state) == 0:

                    state["step"] = 0
                    # Exponential moving average of gradient and squared gradient
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                # for 
                # \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
                # \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                # L_2 weight decay
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Update Adam
                # m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
                # v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Uncostrained step, H metric, Adam denominator
                # psi_t is the adaptive variance estimate

                # denom in ProxAdam.py
                psi_t = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                
                # Intermediate t + 1/2 ---> Adam update: p = p - (lr / denom) * (m_t / bias_correction1)
                phi_t = exp_avg / bias_correction1
                hat_x = p - lr * (phi_t / psi_t)

                # what ProxAdam did once, AdaProx does many times
                if prox_lambda > 0:

                    # Metric diagonalization: H_t = Diag(psi_t)
                    max_psi = psi_t.max()

                    # gamma_t is the sub-problem step size
                    gamma = lr / max_psi 

                    # Initialize z_1 = hat_x
                    z = hat_x.clone() 
                    
                    for _ in range(max_sub_iters):
                        z_prev = z.clone()
                        
                        # Eq12: z_{tau+1} = prox(z - (1/max_psi) * psi * (z - hat_x))
                        # PGM sub-steps 
                        pgm_step = z - (psi_t / max_psi) * (z - hat_x)
                        z = F.softshrink(pgm_step, gamma * prox_lambda)
                        
                        # convergence check
                        if (z - z_prev).norm() < sub_eps * z.norm():
                            break
                    
                    p.copy_(z) 
                else:
                    # no penalty for standard adam
                    p.copy_(hat_x)

        return loss

# Register AdaProx optimizer
info = OptimInfo(name="adaprox", opt_class=AdaProx, description="Official ADAPROX (Melchior et al. 2020)")
default_registry.register(info)