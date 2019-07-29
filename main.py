import matplotlib.pyplot as plt
import time
import sympy
import numpy.linalg as la
import numpy as np
import pickle

from mayavi.mlab import *
from mayavi import mlab

import graph
import lib

ORD = np.inf
X0 = np.array([-0.16050991413641064, 0.49310231154567474])
SYSTEM = ['2 * t1 - sin((t1 - t2) / 2)', '2 * t2 - cos((t1 + t2) / 2)']  # (-0.16050991413641064, 0.49310231154567474)
# SYSTEM = ['1.1 - sin(t2 / 3) + ln(1 + (t1 + t2) / 5) - t1', '0.5 + cos(t1 * t2 / 6) - t2']  # (1.035945755659953, 1.468048170653296), (-6.01497, 1.02064)
variables = sympy.symbols('t1:' + str(len(X0) + 1))
F = sympy.sympify(SYSTEM)
F_lambda = sympy.lambdify(variables, F)

if __name__ == '__main__':
    E = np.identity(len(X0))
    dF = sympy.Matrix(len(F), len(variables), lambda i, j: sympy.diff(F[i], variables[j]))

    for i in range(1,2):
        X0[0] += i * 0.05
        dF_max = la.norm(sympy.lambdify(variables, dF)(*X0), ord=ORD)
        # dF_max = lib.f_max1(lambda t: la.norm(sympy.lambdify(variables, dF)(*t), ord=ORD), X0,
        #                    [(X0[i] - 1, X0[i] + 1) for i in range(len(X0))])

        B = (2 / dF_max) - np.finfo(float).eps
        A = -B

        A = B / 100
        # B = 0.9214638547443138

        # B = 0
        # A = -1.749

        def calc(tau):
            S = lambda x: x - tau * np.array(F_lambda(*x))
            dS = sympy.lambdify(variables, E - tau * dF)
            r = (la.norm(tau * np.array(F_lambda(*X0)), ord=ORD)) / np.finfo(float).eps
            bounds = [(X0[i] - r, X0[i] + r) for i in range(len(X0))]
            q = lib.f_max1(lambda t: la.norm(dS(*t), ord=ORD), X0, bounds)

            if not (0 <= q < 1) or not (la.norm(S(X0) - X0, ord=ORD) <= (1 - q) * r):
                return np.nan
            return q
        x0 = A
        x1 = A + np.finfo(float).eps*20
        y0 = calc(x0)
        y1 = calc(x1)
        fy = lambda _x: (_x - x0) * (y1 - y0) / (x1 - x0) + y0
        fx = lambda _y: (_y - y0) * (x1 - x0) / (y1 - y0) + x0
        B = fx(0) - 0.001

        x = np.linspace(0, 1, 100)
        y=[]
        for B in x:
            y.append(fy(B)-calc(B))

        plt.plot(x, y)
        plt.show()
        # print(fy(B)-calc(B), fy(B), calc(B), max(fy(B), calc(B))/abs(calc(B) - fy(B)))
        # with open('res', 'wb') as f:
        #     pickle.dump(res, f)

        # x, y, z = zip(*res)
        # pts = mlab.points3d(x, y[::-1], z, z, scale_mode='none', scale_factor=0)
        # mesh = mlab.pipeline.delaunay2d(pts)
        # surf = mlab.pipeline.surface(mesh)
        # x1, y1, z1 = zip(*res1)
        # pts1 = mlab.points3d(x1, y1[::-1], z1, z1, scale_factor=0.1)
        #
        # mlab.show()

        # graph.draw_screen(res)

        # lib.star_wars()
        # graph.draw_file(res, 'out4/' + str(i)+'.png')