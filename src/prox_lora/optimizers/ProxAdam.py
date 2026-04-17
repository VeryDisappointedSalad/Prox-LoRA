import torch
from torch.optim import Optimizer
from timm.optim._optim_factory import default_registry
from timm.optim._optim_factory import OptimInfo

class ProxAdam(Optimizer):
    """

    Proximal Adam Optimizer.

    Coding:
    to use in config: OptimizerConfig('opt' = 'prox_adam', ...)
    
    Math:
    1. Adam update to get intermediate theta:
       theta_half = theta_t - (lr / (sqrt(v) + eps)) * m --> Forward

    2. Proximal step for L1 regularization:
       theta_{t+1} = sgn(theta_half) * max(|theta_half| - prox_lambda * lr, 0) --> Backward
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, prox_lambda=0.01, **kwargs):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda}")

        defaults = dict(lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, prox_lambda=prox_lambda)
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

                # Compute adaptive learning rate
                # denom = (sqrt(v_t) / bias_correction2) + eps
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                
                # Intermediate t + 1/2 ---> Adam update: p = p - (lr / denom) * (m_t / bias_correction1)
                # \theta_{t+1/2} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
                step_size = lr / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # Backward prox step
                if prox_lambda > 0:
                    # TODO : is this threshold okay?
                    threshold = lr * prox_lambda
                    
                    # \theta_{t+1} = \text{prox}_{\eta \lambda \|\cdot\|_1}(\theta_{t+1/2}) = \text{sgn}(\theta_{1/2}) \max(|\theta_{1/2}| - \eta \lambda, 0)
                    p.copy_(torch.sign(p) * torch.clamp(p.abs() - threshold, min=0))

        return loss

# Register ProxAdam optimizer
info = OptimInfo(name="ProxAdam", opt_class=ProxAdam, description="Custom Proximal Adam Optimizer")
default_registry.register(info)