import scipy.optimize
from main import X0, F, lambda_from, norm
import numpy as np

np.set_printoptions(precision=None)

TAU = 0.8580172349397682
x = np.array(X0) + [1] * len(X0)

S = lambda _x: _x + TAU * np.array(lambda_from(F)(*_x))
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
print(scipy.optimize.newton_krylov(lambda x: lambda_from(F)(*x), X0))
