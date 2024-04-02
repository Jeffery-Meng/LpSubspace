# Code copied from https://github.com/HIPS/autograd/blob/master/autograd/misc/optimizers.py

"""Some standard gradient-based stochastic optimizers.
These are just standard routines that don't make any use of autograd,
though you could take gradients of these functions too if you want
to do meta-optimization.
These routines can optimize functions whose inputs are structured
objects, such as dicts of numpy arrays."""
import numpy as np
from builtins import range

def adam(grad, x, callback=None, num_iters=100,
         step_size=0.05, b1=0.9, b2=0.999, eps=10**-8):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x)
        # print(np.linalg.norm(g))
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return x