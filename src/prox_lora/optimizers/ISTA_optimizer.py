import torch
from torch.optim import Optimizer


from timm.optim._optim_factory import default_registry
from timm.optim._optim_factory import OptimInfo

class ISTA(Optimizer):
    """
    Coding:
    to use in config: OptimizerConfig('opt' = 'ista', ...)

    Math:
    solves min_theta Loss_function(theta) + lambda * L1_regularization(theta)
    theta' = theta_t - lr *grad_theta Loss_function(theta)
    theta_{t+1} = sgn(theta') * max(|abs(theta')|-prox_lambda * lr, 0)
    """

    def __init__(self, params, lr = 1e-3, momentum = 0, weight_decay = 0, prox_lambda = 0.1, **kwargs):
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr} < 0")
        if prox_lambda < 0:
            raise ValueError(f"Invalid prox_lambda: {prox_lambda} < 0")

        defaults = dict(lr = lr,
                        momentum = momentum,
                        weight_decay = weight_decay,
                        prox_lambda = prox_lambda)

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        #print("ISTA OPTIMIZER IS RUNNING!")
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            lr = group['lr']
            prox_lambda = group['prox_lambda']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                # get the grad
                d_p = p.grad

                # do weight decay if specified, apparently PyTorch normally does this
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # gradient step
                #p' = p - lr * grad
                p.add_(d_p, alpha = -lr)

                # proximal step, soft thresholding for L1
                # prox(w) = sign(w) * max(|w| - lambda * lr, 0)
                if prox_lambda > 0:
                    threshold = prox_lambda * lr
                    p.copy_(torch.sign(p) * torch.maximum(p.abs() - threshold, torch.tensor(0.0, device=p.device)))

        return loss


# Register ISTA optimizer
info = OptimInfo(name='ista', opt_class=ISTA, description="Custom ISTA Optimizer")
default_registry.register(info)