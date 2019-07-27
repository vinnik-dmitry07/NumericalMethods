from typing import Any, Callable, Union

from numpy.core.multiarray import ndarray

from main import X0, F_lambda
import numpy as np

assert -np.log10(np.finfo(np.float).resolution) == 15
np.set_printoptions(precision=16)


TAU = 0.62
R = 0.1
x = X0 + [0.1, 0]

S = lambda _x: _x - TAU * np.array(F_lambda(*_x))

iter_count = 0
while True:
    x_new = S(x)

    if np.allclose(x, x_new, atol=np.finfo(float).eps, rtol=0):
        x = x_new
        break

    x = x_new
    iter_count += 1

print(iter_count, x)
