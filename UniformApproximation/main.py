# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import sympy
from numpy.polynomial import Polynomial
from numpy.polynomial import Chebyshev
from sympy.abc import x
import scipy.optimize


def T(n_):
    return Chebyshev(np.append(np.zeros(n_), 1))


a = 0
b = 2


P_n = Polynomial([0, 1, 0, 1])  # 0 + 1*x + 0*x^2 + 1*x^3
n = P_n.degree()

# Zero approximation
P_n_min = scipy.optimize.fminbound(P_n, a, b, full_output=True)[1]
P_n_max = -scipy.optimize.fminbound(-P_n, a, b, full_output=True)[1]
Q0 = (P_n_max + P_n_min) / 2

# First approximation
alpha1 = (P_n(b) - P_n(a)) / (b - a)
r = (P_n.deriv() - Polynomial([alpha1])).roots()
d = next(t for t in r if a < t < b)
alpha0 = (P_n(a) + P_n(d) - alpha1 * (a + d)) / 2
Q1 = alpha0 + alpha1 * x
Q1 = sympy.lambdify(x, Q1)

# Second approximation
Q2 = P_n(x) - P_n.coef[n] * T(n)((2 * x - (a + b)) / (b - a)) * ((b - a) ** n) / (2 ** (2 * n - 1))
Q2 = sympy.lambdify(x, Q2)

X = np.linspace(0, 2, 1000)

plt.plot(X, X + X ** 3, color='tab:blue', label='x + x³')
plt.axhline(Q0, color='tab:orange', label='Zero approximation')
plt.plot(X, Q1(X), color='tab:green', label='First approximation')
plt.plot(X, Q2(X), color='tab:red', label='Second approximation')

plt.legend(loc='upper left')
plt.show()
