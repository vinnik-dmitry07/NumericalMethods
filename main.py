import time
import sympy
import numpy.linalg as la
import numpy as np
import pickle
import graph
import lib


ORD = np.inf
X0 = np.array([-0.16050991413641064, 0.49310231154567474])
N = 40
SYSTEM = ['2 * t1 - sin((t1 - t2) / 2)', '2 * t2 - cos((t1 + t2) / 2)']  # (-0.16050991413641064, 0.49310231154567474)
# SYSTEM = ['1.1 - sin(t2 / 3) + ln(1 + (t1 + t2) / 5) - t1', '0.5 + cos(t1 * t2 / 6) - t2']  # (1.035945755659953, 1.468048170653296), (-6.01497, 1.02064)
variables = sympy.symbols('t1:' + str(len(X0) + 1))
F = sympy.sympify(SYSTEM)
F_lambda = sympy.lambdify(variables, F)

if __name__ == '__main__':
    E = np.identity(len(X0))
    dF = sympy.Matrix(len(F), len(variables), lambda i, j: sympy.diff(F[i], variables[j]))

    for i in range(3, 4):
        X0[0] += i * 0.05
        # dF_max = la.norm(sympy.lambdify(variables, dF)(*X0), ord=ORD)
        dF_max = lib.f_max1(lambda t: la.norm(sympy.lambdify(variables, dF)(*t), ord=ORD), X0,
                           [(X0[i] - 1, X0[i] + 1) for i in range(len(X0))])
        dF_min = lib.f_max1(lambda t: -la.norm(sympy.lambdify(variables, dF)(*t), ord=ORD), X0,
                           [(X0[i] - 1, X0[i] + 1) for i in range(len(X0))])
        #print(2/(abs(dF_max)+abs(dF_min)))
        B = 2 / dF_max
        A = -B

        A = 0
        # B = 0.9214638547443138

        # B = 0
        # A = -1.749

        step = (B - A) / N
        t1=[]
        res = []
        for tau in np.linspace(A + step, B - step, N):
            for r in [0.221647987687213]: #np.linspace(0, 1, N):  # [0.070708]

                bounds = [(X0[i] - r, X0[i] + r) for i in range(len(X0))]

                S = lambda x: x - tau * np.array(F_lambda(*x))

                dS = sympy.lambdify(variables, E - tau * dF)
                q = lib.f_max1(lambda t: la.norm(dS(*t), ord=ORD), X0, bounds)
                t1.append(la.norm(S(X0) - X0) / (1 - q))
                if not (0 <= q < 1):
                    res.append((tau, r, np.nan))
                    continue
                if not (la.norm(S(X0) - X0, ord=ORD) <= (1 - q) * r):
                    res.append((tau, r, np.nan))
                    continue

                res.append((tau, r, q))
        print(res)
        print(min(t1))
        with open('res', 'wb') as f:
            pickle.dump(res, f)

        graph.draw_screen(res)

        # lib.star_wars()
        # graph.draw_file(res, 'out4/' + str(i)+'.png')
