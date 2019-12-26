import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.chebyshev as cheb
import sympy
from sympy import Product, Sum, DeferredVector, lambdify
from sympy.abc import a, j
from sympy.tensor import IndexedBase, Idx

N = 4
A = -3
B = 3
x = IndexedBase('x')
y = IndexedBase('y')

i = sympy.symbols('i', cls=Idx)

L = Sum(
    y[i] *
    Product((a - x[j]) / (x[i] - x[j]), (j, 0, i - 1)).doit() *
    Product((a - x[j]) / (x[i] - x[j]), (j, i + 1, N)).doit(),
    (i, 0, N)
).doit()
L_lambda = lambdify([a, DeferredVector('x'), DeferredVector('y')], L, 'numpy')

display_grid = np.linspace(A, B, 1000)
interpol_grid = np.linspace(A, B, N + 1)
che_interpol_grid = (cheb.chebroots((0, 0, 0, 0, 0, 1)) * (B - A) + A + B) / 2
plt.plot(display_grid, L_lambda(a=display_grid, x=interpol_grid, y=np.sin(interpol_grid)), color='red',
         label='Lagrange polynomial (equidistant nodes)')
plt.plot(display_grid, L_lambda(a=display_grid, x=che_interpol_grid, y=np.sin(che_interpol_grid)), color='blue',
         label='Lagrange polynomial (roots-of-Chebyshev-series nodes)')
plt.plot(display_grid, np.sin(display_grid), color='green', label='sin(x)')

plt.legend(loc='upper left')
plt.show()
