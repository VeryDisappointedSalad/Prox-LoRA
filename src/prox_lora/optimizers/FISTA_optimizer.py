import torch
from torch.optim import Optimizer

#from timm.optim._optim_factory import default_registry

#import timm.optim.optim_factory
#import timm.optim._optim_factory as internal_factory ### _optim_factory instead of optim_factory
#from timm.optim.optim_factory import default_registry
from timm.optim._optim_factory import default_registry
from timm.optim._optim_factory import OptimInfo

#@default_registry.register
class FISTA(Optimizer):

    """
    Coding:
    to use in config: OptimizerConfig('opt' = 'fista', ...)

    Math:
    solves min_theta Loss_function(theta) + lambda * L1_regularization(theta)

    y_{t+1} = prox(x_t - lr * grad)
    y_{t+1} = sgn(x_t) * max(|abs(x_t)|-prox_lambda * lr, 0)
    x_{t+1} = y_{t+1} + ((t-1)/(t+2)) * (y_{t+1} - y_t)
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
        #print("FISTA OPTIMIZER IS RUNNING!")
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

                # initialize counter and state
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 1
                    state['y_prev'] = p.clone().detach() # the y_t

                # get current state
                t = state['t']
                y_prev = state['y_prev']

                # do weight decay if specified, apparently PyTorch normally does this
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # gradient step from x_t
                #p' = p - lr * grad
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
                state['y_prev'] = y_next
                state['t'] +=1

        return loss


info = OptimInfo(name='fista', opt_class=FISTA, description="Custom FISTA Optimizer")
default_registry.register(info)


#if hasattr(timm.optim.optim_factory, '_OPTIMIZERS'):
#    timm.optim.optim_factory._OPTIMIZERS['fista'] = FISTA
#    print(f"Successfully registered 'prox_sgd'")
#    #default_registry.register('fista', ProximalSGD)

#internal_factory._OPTIMIZERS['fista'] = FISTA


#if 'fista' in default_registry:
#    del default_registry['fista'] # Clear if exists to avoid conflicts
   
#default_registry.register('fista', FISTA)
#print('Registered FISTA!')