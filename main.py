from collections import namedtuple

import matplotlib.pyplot as plt

import sympy
import numpy.linalg as la
import numpy as np
# import pickle
# import graph
import lib

Point = namedtuple('Point', ['x', 'y'])


def lambda_from(expr):
    return sympy.lambdify(VARIABLES, expr)


def norm(x_):
    return la.norm(x_, ord=ORD)


def collinear(p1: Point, p2: Point, p3: Point):
    if not (p3.y - p2.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p2.x) < EPS:
        print(p1, p2, p3)
    return (p3.y - p2.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p2.x) < EPS


def line_from(p1: Point, p2: Point):
    return lambda x: (x - p1.x)*(p2.y-p1.y)/(p2.x-p1.x) + p1.y


def onclick(event):
    print(event.xdata, event.ydata)


def super_space(N):
    x = N // 2
    yield x

    r = (N // 2) - 1 if N % 4 == 0 else N // 2
    for n in range(1, r):
        d = n * 2
        if (n + 1) % 2 == 0:
            d = -d
        x += d
        yield x
    yield 0

    if N % 4 == 0:
        x = N - 1
    else:
        x = N - 2
    yield x

    for n in range(r + 2, N):
        d = (N - n) * 2
        if (n + 1) % 2 == 0:
            d = -d
        x += d
        yield x


EPS = np.finfo(float).eps
ORD = np.inf
# X0 = [-0.16050991413641064, 0.49310231154567474]
# SYSTEM = ['2 * t1 - sin((t1 - t2) / 2)', '2 * t2 - cos((t1 + t2) / 2)']  # 0.6148771368394607
X0 = [1.035945755659953, 1.468048170653296]
SYSTEM = ['1.1 - sin(t2 / 3) + ln(1 + (t1 + t2) / 5) - t1', '0.5 + cos(t1 * t2 / 6) - t2']  # -1.0959715096866944

# X0 = [1, 0.95]
# SYSTEM = ['t1^2 - 2 * t2^2 - t1*t2 + 2*t1 - t2 + 1', '2 * t1^2 - t2^2 + t1*t2 + 3*t2 - 5']  # 0.016712364538451496

VARIABLES = sympy.symbols('t1:' + str(len(X0) + 1))
F = sympy.sympify(SYSTEM)
N = 10000
assert N % 2 == 0 and N >= 6

if __name__ == '__main__':
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    E = np.identity(len(X0))
    dF = sympy.Matrix(len(F), len(VARIABLES), lambda i_, j_: sympy.diff(F[i_], VARIABLES[j_]))

    for i in range(2, 3):
        X0[0] += i * 0.05

        opt1 = norm(np.array(lambda_from(F)(*X0))) / EPS

        def calc(tau):
            dS = E - tau * dF
            r = abs(tau) * opt1
            bounds = [(X0[b] - r, X0[b] + r) for b in range(len(X0))]
            q = lib.f_max(lambda t: norm(lambda_from(dS)(*t)), X0, bounds)
            return q if 0 <= q < 1 else np.nan


        h = 2 / lib.f_max(lambda t: norm(lambda_from(dF)(*t)), X0)
        # h = 2 / norm(lambda_from(dF)(*X0))
        # min_ = 1.2553410768508908e-16
        # max_ = 0.90309207717
        print(h)
        # x1 = np.linspace(6e-16, 1e-2, 3)
        # t = (calc(x1[2])-calc(x1[0])) / (x1[2] - x1[0])
        # y1=[]
        # for x_i in x1:
        #     y1.append(calc(x_i))
        # print(collinear(x1[0], y1[0], x1[1], y1[1], x1[2], y1[2]))
        # fq = lambda _tau: _tau * t + 1
        # a, b = 0.00883884633627, 0.00883884633629
        # a, b = 0.5122545421104, 0.51225454211041
        # a = -3.762730595697267e-16
        # b = 3.762730595697267e-16
        a = -1.0959
        b = 0

        # space = np.linspace(-h, -h+0.1, 40)
        # i = 0
        # for x in super_space(40):
        #     if x != 0 and x != 40 - 1:
        #         points = []
        #         for t in (x-1, x, x+1):
        #             x_i = space[t]
        #             y_i = calc(x_i)
        #             if np.isnan(y_i):
        #                 break
        #             points.append(Point(x_i, y_i))
        #         if len(points) < 3:
        #             continue
        #         else:
        #             print(points)
        #         i += 1
        #         print(i)
        #         if not collinear(*points):
        #             print(123)
        points = [Point(x=-1.743596702592648, y=0.9512197021381225), Point(x=-1.7410326000285454, y=0.9483502728521295), Point(x=-1.7384684974644429, y=0.9454808435661369)]
        points1 = [Point(x=-0.8, y=0.43537036417398506), Point(x=-0.5125348393130676, y=0.6382595504131462),
                 Point(x=-0.22506967862613525, y=0.8411487366523088)]
        l1 = line_from(points1[0], points1[2])
        l = line_from(points[0], points[2])
        # b = a + int((b - a) / EPS) * EPS
        x = np.linspace(a, b, N)
        print(x[1] - x[0], x[1] - x[0] >= EPS)
        y = []
        y1 = []
        y2 = []
        ca = 0
        cb = 0
        for x_i in x:
            t1 = calc(x_i)
            if not np.isnan(t1):
                ca += 1
                if l1(x_i) - t1 > EPS:
                    cb += 1
            y.append(t1)
            y1.append(l(x_i))
            y2.append(l1(x_i))
        print(cb, ca)
        # plt.plot(x, y)
        # plt.plot((np.array(x) - a) / EPS, y)

        # plt.plot(x, y1)
        # plt.plot(x, y2)

        # plt.axes().set_aspect('equal', 'datalim')

        # plt.axvline(x=h)
        # plt.axvline(x=-h)
        # plt.ylim(0, 1)

        # plt.show()

        # with open('res', 'wb') as f:
        #     pickle.dump(res, f)

        # graph.draw_screen(res)

        # lib.star_wars()
