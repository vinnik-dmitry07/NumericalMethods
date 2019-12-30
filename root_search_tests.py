import timeit

setup = '''\
from scipy import optimize
from main import ORD
import numpy as np
import numpy.linalg as la


def elem_f(t1, t2):
    return la.norm(
        [[0.00912156139039826*np.cos((1/2)*t1 - 1/2*t2) + 0.963513754438407, - 0.00912156139039826*np.cos((1/2)*t1 - 1/2*t2)], 
         [-0.00912156139039826*np.sin((1/2)*t1 + 1/2*t2), 0.963513754438407 - 0.00912156139039826*np.sin((1/2)*t1 + (1/2)*t2)]], 
    ord=ORD)


x0 = [1, 1]
r = 1

bounds = []
for x0_i in x0:
    bounds.append((x0_i - r, x0_i + r))
'''

tests = (
    "elem_f(*optimize.fmin_tnc(func=lambda t: -elem_f(*t), x0=x0, approx_grad=True, bounds=bounds, disp=0)[0])",
    "-optimize.fmin_l_bfgs_b(func=lambda t: -elem_f(*t), x0=x0, approx_grad=True, bounds=bounds)[1]",
    "-optimize.fmin_slsqp(func=lambda t: -elem_f(*t), x0=x0, bounds=bounds, iprint=0, full_output=True)[1]",
    "-optimize.fmin_l_bfgs_b(func=lambda t: -elem_f(*t), x0=x0, approx_grad=True, bounds=bounds, epsilon=np.finfo(float).eps)[1]"
)

for test in tests:
    print(timeit.timeit(test, setup=setup, number=20000))
