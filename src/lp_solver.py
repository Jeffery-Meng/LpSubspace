import numpy as np
from optimizer import adam


def objective_func(A, x, y, LP):
    x = x.reshape((-1, 1))
    res = A@x - y
    res = res.reshape((-1))
    return np.linalg.norm(res, LP)

def gradient_func(A, x, y, LP):
    x = x.reshape((-1, 1))
    vec = A @ x - y
    vec = np.power(np.abs(vec), LP - 1) * np.sign(vec)
    result = LP * vec.transpose() @ A
    result = result.reshape((-1))
    return result

def compute_lp(A, y, LP):
    A = (A/1000).T
    arows, acols = A.shape
    xx = np.zeros(acols)
    x_star = adam(lambda var: gradient_func(A, var, y, LP), xx, num_iters=400)
    return objective_func(A, x_star, y, LP)
