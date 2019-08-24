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
    return lambda x_: (x_ - p1.x)*(p2.y-p1.y)/(p2.x-p1.x) + p1.y


def lines_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
    denominator = ((p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x))
    return Point(
        ((p1.x*p2.y - p1.y*p2.x)*(p3.x - p4.x) - (p1.x - p2.x)*(p3.x*p4.y - p3.y*p4.x)) / denominator,
        ((p1.x*p2.y - p1.y*p2.x)*(p3.y - p4.y) - (p1.y - p2.y)*(p3.x*p4.y - p3.y*p4.x)) / denominator
    )


def onclick(event):
    print(event.xdata, event.ydata)


def nearest_floor_even(x):
    return int(np.floor(x / 2) * 2)


def super_space(N):
    assert N % 2 == 0 and N >= 6
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

# X0 = [1.035945755659953, 1.468048170653296]
# SYSTEM = ['1.1 - sin(t2 / 3) + ln(1 + (t1 + t2) / 5) - t1', '0.5 + cos(t1 * t2 / 6) - t2']  # -1.0959715096866944

# X0 = [-5/3, -1/3]  # X0[0] = 0
# X0 = [-3/2, 1/2]
# X0 = [1, 1]
# SYSTEM = ['t1^2 - 2 * t2^2 - t1*t2 + 2*t1 - t2 + 1', '2 * t1^2 - t2^2 + t1*t2 + 3*t2 - 5']  # 0.016712364538451496

# X0 = [3.4874427876429534523, 2.261628630553593956]  # - don't work
# X0 = [1.4588902301521780083, -1.396767009181618128]
# SYSTEM = ['2*t1^2 - t1*t2 - 5*t1 + 1', 't1 + 3*log(t1, 10) - t2^2']  # ~0.36

# X0 = [1, 2]
# X0 = [-1, -2]
# SYSTEM = ['3*t1^2*t2^2 + t1^2 - 3*t1*t2 - 7', 't1*t2 - 2']  # - bad

# X0 = [1, 1]
# X0 = [-9/5, -3/5]
# SYSTEM = ['(t1 + 2*t2)*(2*t1 - t2 + 1) - 6', '(2*t1 - t2 + 1)/(t1 + 2*t2) - 2/3']  # - bad

# X0 = [0, 3]
# SYSTEM = ['t1 + t2 - 3', 't1^2 + t2^2 - 9']  # - bad

# X0 = [-0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
X0 = [0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
SYSTEM = ['t1^2 + t2^2 + t3^2 - 1', '2*t1^2 + t2^2 - 4*t3', '3*t1^2 - 4*t2 + t3^2']  # - bad

VARIABLES = sympy.symbols('t1:' + str(len(X0) + 1))
F = sympy.sympify(SYSTEM)
N = 1000
assert N % 2 == 0 and N >= 6

if __name__ == '__main__':
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    E = np.identity(len(X0))
    dF = sympy.Matrix(len(F), len(VARIABLES), lambda i_, j_: sympy.diff(F[i_], VARIABLES[j_]))

    # X0[0] += 0.5

    opt1 = norm(np.array(lambda_from(F)(*X0))) / EPS

    def calc(tau, no_bounds=False):
        dS = E - tau * dF
        if not no_bounds and norm(lambda_from(dS)(*X0)) >= 1:
            return np.nan
        r = abs(tau) * opt1
        bounds = [(X0[b] - r, X0[b] + r) for b in range(len(X0))]
        q = lib.f_max(lambda t: norm(lambda_from(dS)(*t)), X0, bounds)
        return q if 0 <= q < 1 or no_bounds else np.nan


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
    a = b = None
    i = 0
    for t in super_space(nearest_floor_even(h/EPS)):
        if not np.isnan(calc(t*EPS)):
            a = 0
            b = h
            break
        elif not np.isnan(calc(-t*EPS)):
            a = -h
            b = 0
            break
        i += 1
        if i > 1000:
            raise Exception()
    if a is None:
        raise Exception()

    l = None
    d = h
    while True:
        x = np.linspace(a, a + d, 5)[1:-1]
        y = np.vectorize(calc)(x)
        points = [Point(x_i, y_i) for x_i, y_i in zip(x, y)]
        if not np.any(np.isnan(y)) and points[0].y > points[2].y and collinear(*points):
            l = line_from(points[0], points[2])
            break
        else:
            d /= 2
            if d < 2*EPS:
                break
    print(d)
    l1 = None
    d = h
    while True:
        x = np.linspace(b-d, b, 5)[1:-1]
        y = np.vectorize(calc)(x)
        points1 = [Point(x_i, y_i) for x_i, y_i in zip(x, y)]
        if not np.any(np.isnan(y)) and points1[0].y < points1[2].y and collinear(*points1):
            l1 = line_from(points1[0], points1[2])
            break
        else:
            d /= 2
            if d < 2*EPS:
                break
    # g = super_space(10)
    # while True:
    #     t1 = next(g)
    #     t2 = next(g)
    #     t3 = next(g)

    if l is not None and l1 is not None:
        intersection = lines_intersection(points[0], points[2], points1[0], points1[2])
    elif l is not None:
        d = 0
        while True:
            y = calc(b - d)
            if not np.isnan(y):
                break
            d += EPS
        intersection = Point(b - d, y)
    elif l1 is not None:
        d = 0
        while True:
            y = calc(a + d)
            if not np.isnan(y):
                break
            d += EPS
        intersection = Point(a + d, y)
    else:
        raise Exception()

    print(intersection.x, calc(intersection.x))
    plt.scatter(intersection.x, intersection.y)

    x = np.linspace(a, b, N + 1)
    y3 = []
    for x_i in x:
        y3.append(1 + (x_i if x_i < 0 else -x_i))

    plot = plt.scatter(x, np.vectorize(calc)(x, False))
    if l is not None:
        plt.plot(x, l(x))
    if l1 is not None:
        plt.plot(x, l1(x))
    # plt.plot(x, x)

    # plt.axes().set_aspect('equal', 'datalim')

    # plt.axvline(x=h)
    # plt.axvline(x=-h)

    # x1 = np.linspace(0, h / 1e8, 10)
    # y1 = np.vectorize(calc)(x1)
    # x1, y1 = zip(*((x_i, y_i) for x_i, y_i in zip(x1, y1) if not np.isnan(y_i)))
    #
    # y_r = (lambda x_i: np.polyval(np.polyfit(x1, y1, 1), x_i))(x)
    # plt.plot(x, y_r)
    # plt.scatter(x, y-y_r, s=1)

    plt.ylim(0, 1)
    plt.xlim(a, b)
    plt.show()

    # with open('res', 'wb') as f:
    #     pickle.dump(res, f)

    # graph.draw_screen(res)

    # lib.star_wars()
