from typing import Any, Callable, Union

from numpy.core.multiarray import ndarray

from main import X0, F, lambda_from, norm, EPS
import numpy as np

assert -np.log10(np.finfo(np.float).resolution) == 15
np.set_printoptions(precision=16)

TAU = 0.21224502320231167
x = X0 + np.array([0.01,0.01])

S = lambda _x: _x - TAU * np.array(lambda_from(F)(*_x))
q = 0.7142857142857142
r = norm(S(X0) - X0) / (1 - q)
print(max(0, np.floor(np.log(0.001 / (2 * r)) / np.log(q)) + 1))


iter_count = 0
while True:
    x_new = S(x)

    if np.allclose(x, x_new, atol=0.001, rtol=0):
        x = x_new
        break

    x = x_new
    iter_count += 1

print(iter_count, x)
