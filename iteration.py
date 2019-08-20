from typing import Any, Callable, Union

from numpy.core.multiarray import ndarray

from main import X0, F, lambda_from, norm, EPS
import numpy as np

assert -np.log10(np.finfo(np.float).resolution) == 15
np.set_printoptions(precision=16)


TAU = 0.016712364538451496
R = abs(TAU) * norm(np.array(lambda_from(F)(*X0))) / EPS
x = X0

S = lambda _x: _x - TAU * np.array(lambda_from(F)(*_x))

iter_count = 0
while True:
    x_new = S(x)

    if np.allclose(x, x_new, atol=EPS, rtol=0):
        x = x_new
        break

    x = x_new
    iter_count += 1

print(iter_count, x)
