import torch
from torch.optim import Optimizer
import timm.optim.optim_factory
#from timm.optim.optim_factory import default_registry

class ProximalSGD(Optimizer):

    """
    Coding:
    My first custom Proximal-esque optimizer.
    to use in config: OptimizerConfig('opt' = 'prox_sgd', lr = 0.01)

    Math:
    solves min_theta Loss_function(theta) + lambda * L1_regularization(theta)
    theta' = theta_t - lr *grad_theta Loss_function(theta)
    theta_{t+1} = sgn(theta') * max(|abs(theta')|-prox_lambda * lr, 0)
    """

    def __init__(self, params, lr = 1e-3, momentum = 0, weight_decay = 0, prox_lambda = 0.1, **kwargs):
        
        defaults = dict(lr = lr,
                        momentum = momentum,
                        weight_decay = weight_decay,
                        prox_lambda = prox_lambda)

        super().__init__(params, defauls)

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            prox_lambda = group['prox_lambda']
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # get the grad
                d_p = p.grad

                # do weight decay if specified
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                
                # do momentum if specified
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                # gradient step
                #p' = p - lr * grad
                p.add_(d_p, alpha = -lr)

                # proximal step, soft thresholding for L1
                # prox(w) = sign(w) * max(|w| - lambda * lr, 0)
                if prox_lambda > 0:
                    threshold = prox_lambda * lr
                    p.copy_(torch.sign(p) * torch.maximum(p.abs() - threshold, torch.tensor(0.0, device=p.device)))

        return loss


if hasattr(timm.optim.optim_factory, '_OPTIMIZERS'):
    #default_registry.register('fista', ProximalSGD)
    timm.optim.optim_factory._OPTIMIZERS['prox_sgd'] = ProximalSGD
    print(f"Successfully registered 'prox_sgd'")