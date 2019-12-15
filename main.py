import matplotlib.pyplot as plt

import sympy
import numpy.linalg as la
import numpy as np
import lib


def lambda_from(expr):
    return sympy.lambdify(VARIABLES, expr)


def norm(x_):
    return la.norm(x_, ord=ORD)


def onclick(event):
    print(event.xdata, event.ydata)


EPS = np.finfo(float).eps
ORD = np.inf

# X0 = [-0.16050991413641064, 0.49310231154567474]
# SYSTEM = ['2 * t1 - sin((t1 - t2) / 2)', '2 * t2 - cos((t1 + t2) / 2)']  # 0.6148771368394607

X0 = [1.0359457556599534, 1.468048170653296]
SYSTEM = ['1.1 - sin(t2 / 3) + ln(1 + (t1 + t2) / 5) - t1', '0.5 + cos(t1 * t2 / 6) - t2']  # -1.0959715096866944

# X0 = [-5/3, -1/3]; X0[0] = 0
# X0 = [-3/2, 1/2]
# X0 = [1, 1]
# SYSTEM = ['t1^2 - 2 * t2^2 - t1*t2 + 2*t1 - t2 + 1', '2 * t1^2 - t2^2 + t1*t2 + 3*t2 - 5']  # 0.016712364538451496

# X0 = [3.4874427876429534523, 2.261628630553593956]  # - don't work
# X0 = [1.4588902301521780083, -1.3967670091816181275121]
# SYSTEM = ['2*t1^2 - t1*t2 - 5*t1 + 1', 't1 + 3*log(t1, 10) - t2^2']  # ~0.36

# X0 = [0.0128241458299864, -0.1778006679626201, 0.2446880443442363]
# SYSTEM = ['t1 + t1^2 -2*t2*t3 - 0.1', 't2 - t2^2 + 3*t1*t3 + 0.2', 't3 + t3^2 + 2*t1*t2 - 0.3']  # tau=1.0000000000000004 q=4.44089209850063e-16

# X0 = [1, 2]
# X0 = [-1, -2]
# SYSTEM = ['3*t1^2*t2^2 + t1^2 - 3*t1*t2 - 7', 't1*t2 - 2']  # - bad

# X0 = [1, 1]
# X0 = [-9/5, -3/5]
# SYSTEM = ['(t1 + 2*t2)*(2*t1 - t2 + 1) - 6', '(2*t1 - t2 + 1)/(t1 + 2*t2) - 2/3']  # - bad

# X0 = [0, 3]
# SYSTEM = ['t1 + t2 - 3', 't1^2 + t2^2 - 9']  # - bad

# X0 = [-0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
# X0 = [0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
# SYSTEM = ['t1^2 + t2^2 + t3^2 - 1', '2*t1^2 + t2^2 - 4*t3', '3*t1^2 - 4*t2 + t3^2']  # - bad

# X0 = [1.087981660972634, 2.6239220287034635, 2.1427389523525737]
# SYSTEM = ['log(t2/t3, 10) + 1 - t1', '0.4 + t3^2 - 2*t1^2 - t2', '2 + t1*t2/20 - t3']  # - bad

# X0 = [-0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
# X0 = [0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
# SYSTEM = ['t1^2 + t2^2 + t3^2 - 1', '2*t1^2 + t2^2 - 4*t3', '3*t1^2 - 4*t2 + t3^2']  # - bad

# X0 = [0.2116867939228672, -0.0822360732372419, -0.2621347876899104, 0.2922477600311648]
# SYSTEM = ['8*t1 - t2 - 2*t3 - 2.3', '10*t2 + t3 + 2*t4 + 0.5', '-t1 + 6*t3 + 2*t4 + 1.2',
#           '3*t1 - t2 + 2*t3 + 12*t4 - 3.7']

VARIABLES = sympy.symbols('t1:' + str(len(X0) + 1))
F = sympy.sympify(np.array(SYSTEM))
np.set_printoptions(precision=None)

N = 100
assert N % 2 == 0 and N >= 6

if __name__ == '__main__':
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    E = np.identity(len(X0))
    dF = sympy.Matrix(len(F), len(VARIABLES), lambda i_, j_: sympy.diff(F[i_], VARIABLES[j_]))

    r = 1.5
    b = [(X0_i - r, X0_i + r) for X0_i in X0]
    X0 = np.array(X0) + [1] * len(X0)

    diag = np.array([lambda_from(dF[j, j])(*X0) for j in range(len(X0))])
    if all(diag > 0):
        F = -F
        dF = -dF
        diag = -diag
    elif not all(diag < 0):
        exit(1)
    non_diag = np.sum(
        np.array([[abs(lambda_from(dF[k, j])(*X0)) for j in range(len(X0)) if j != k] for k in range(len(X0))]), axis=1)
    u = np.concatenate((-diag + non_diag, -diag - non_diag))

    if not all(u > 0):
        exit(2)
    tau = 2 / (u.max() + u.min())
    q = (u.max() - u.min()) / (u.max() + u.min())
    print(tau, q)


    def calc(tau, ignore=False):
        dS = E + tau * dF
        q = lib.f_max(lambda x: norm(lambda_from(dS)(*x)), X0, b)
        return q if 0 <= q < 1 or ignore else np.nan

    dF_X0_inv = np.linalg.inv(lambda_from(dF)(*X0))
    print(calc(-dF_X0_inv))  # Calculate q with tau obtained from Newton method
    print(calc(tau))  # Calculate q with tau obtained from our method
    print(calc(tau) / calc(-dF_X0_inv))
    x = np.linspace(0, tau * 2, N + 1)
    y = np.vectorize(calc)(x)
    plot = plt.scatter(x, y, s=6)
    plt.show()
