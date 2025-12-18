import torch
from torch.optim import Optimizer
from timm.optim._optim_factory import default_registry
from timm.optim._optim_factory import OptimInfo

from typing import Any, Dict, Optional

class ADMM(Optimizer):

    """
    Coding: 
    to use in config: OptimizerConfig('opt' = 'admm', lr = 0.01, prox_lambda = 0.1, rho = 0.001)

    Math:
    solves min_theta Loss_function(theta) + lambda * L1_regularization(z) for theta = z

    Lagrangrian:
    L_rho(theta, z, u) = Loss_function(theta) + lambda * L1_regularization(z) + (rho/2) * ||theta - z + u||_2^2
    """

    def __init__(self, params, lr = 1e-3, momentum = 0, weight_decay = 0, prox_lambda = 0.001, rho = 0.001, **kwargs):

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr} < 0")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda} < 0")
        if rho <= 0:
            raise ValueError(f"Invalid rho: {rho} (must be > 0)")

        defaults = dict(lr = lr,
                        momentum = momentum,
                        weight_decay = weight_decay,
                        prox_lambda = prox_lambda,
                        rho = rho)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            prox_lambda = group['prox_lambda']
            rho = group['rho']
            lr = group['lr']
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # initialize state
                state = self.state[p]
                if len(state) == 0:

                    # z_0 = theta_0
                    state['z'] = p.clone().detach()

                    # u_0 = 0 
                    state['u'] = torch.zeros_like(p)

                    # momentum buffer for theta-step if any given
                    if momentum > 0:
                        state['momentum_buffer'] = torch.zeros_like(p)

                # initialize variables
                z = state['z']
                u = state['u']
                
                # grad = d_p = d(Loss)/d(theta)
                d_p = p.grad

                # theta_{k+1} = argmin (Loss(theta) + (rho/2)||theta - z_k + u_k||^2)
                # one gradient secent step
                # grad of penalty = rho * (theta - z + u)
                
                penalty_grad = rho * (p - z + u)
                full_grad = d_p + penalty_grad
                
                if weight_decay != 0:
                    full_grad.add_(p, alpha=weight_decay)

                if momentum > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(full_grad)
                    full_grad = buf

                # gradient step: theta = theta - lr * full_grad
                p.add_(full_grad, alpha=-lr)


                # z-sparsity
                # update -> z_{k+1} = argmin (lambda|z|_1 + (rho/2)||theta_{k+1} - z + u_k||^2)
                # solution to update -> z_{k+1} = Prox_{lambda/rho} (theta_{k+1} + u_k)
                # the threshold is lambda / rho
                
                # target for proximal operator: v = theta_{k+1} + u_k
                v = p + u
                
                # threshold calculation
                if rho > 0 and prox_lambda > 0:
                    threshold = prox_lambda / rho
                    
                    # soft thresholding: sign(v) * max(|v| - threshold, 0)
                    z_new = torch.sign(v) * torch.maximum(v.abs() - threshold, torch.tensor(0.0, device=p.device))
                else:
                    z_new = v # No sparsity if lambda=0, because it zero's out the z

                # Update state 'z'
                state['z'] = z_new


                # dual part, U-update
                # u_{k+1} = u_k + (theta_{k+1} - z_{k+1})
                
                # u = u + (p - z_new)
                u.add_(p - z_new)
                
                # update u
                state['u'] = u

        return loss

# Register ADMM optimizer
info = OptimInfo(name='admm', opt_class=ADMM, description="Custom ADMM Optimizer")
default_registry.register(info)