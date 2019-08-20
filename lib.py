import pickle
import sys
import time
import winsound
import numpy as np
from scipy import optimize


def f_max(func, x0, bounds=None):
    return -optimize.fmin_l_bfgs_b(func=lambda *t: -func(*t), x0=x0, approx_grad=True, bounds=bounds, factr=1.0,
                                   pgtol=np.finfo(float).eps, maxfun=np.inf, maxiter=np.inf,
                                   epsilon=np.finfo(float).eps, maxls=np.iinfo(np.int).max)[1]


def closed_ball(x, r):
    res = []
    for x_i in x:
        res.append((x_i - r, x_i + r))
    return res


def star_wars():
    with open("star_wars", "rb") as f:
        x = pickle.load(f)
        for fq, lt, dr in x:
            winsound.Beep(fq, lt)
            time.sleep(dr)
