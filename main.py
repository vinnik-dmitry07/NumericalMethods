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
    return abs((p3.y - p2.y) * (p2.x - p1.x) - (p2.y - p1.y) * (p3.x - p2.x)) < EPS


def line_from(p1: Point, p2: Point):
    return lambda x_: (x_ - p1.x) * (p2.y - p1.y) / (p2.x - p1.x) + p1.y


def lines_intersection(p1: Point, p2: Point, p3: Point, p4: Point):
    denominator = ((p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x))
    return Point(
        ((p1.x * p2.y - p1.y * p2.x) * (p3.x - p4.x) - (p1.x - p2.x) * (p3.x * p4.y - p3.y * p4.x)) / denominator,
        ((p1.x * p2.y - p1.y * p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x * p4.y - p3.y * p4.x)) / denominator
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

X0 = [-0.16050991413641064, 0.49310231154567474]
SYSTEM = ['2 * t1 - sin((t1 - t2) / 2)', '2 * t2 - cos((t1 + t2) / 2)']  # 0.6148771368394607

# X0 = [1.035945755659953, 1.468048170653296]
# SYSTEM = ['1.1 - sin(t2 / 3) + ln(1 + (t1 + t2) / 5) - t1', '0.5 + cos(t1 * t2 / 6) - t2']  # -1.0959715096866944

# X0 = [-5/3, -1/3]; X0[0] = 0
# X0 = [-3/2, 1/2]
# X0 = [1, 1]
# SYSTEM = ['t1^2 - 2 * t2^2 - t1*t2 + 2*t1 - t2 + 1', '2 * t1^2 - t2^2 + t1*t2 + 3*t2 - 5']  # 0.016712364538451496

# X0 = [3.4874427876429534523, 2.261628630553593956]  # - don't work
# X0 = [1.4588902301521780083, -1.3967670091816181275121]
# SYSTEM = ['2*t1^2 - t1*t2 - 5*t1 + 1', 't1 + 3*log(t1, 10) - t2^2']  # ~0.36

# X0 = [0, 0, 0]
# SYSTEM = ['t1 + t1^2 -2*t2*t3 - 0.1', 't2 - t2^2 + 3*t1*t3 + 0.2', 't3 + t3^2 + 2*t1*t2 - 0.3']  # tau=1.0000000000000004 q=4.44089209850063e-16

# X0 = [1, 2]
# X0 = [-1, -2]  # Need -
# SYSTEM = ['3*t1^2*t2^2 + t1^2 - 3*t1*t2 - 7', 't1*t2 - 2']  # - bad

# X0 = [1, 1]
# X0 = [-9/5, -3/5]
# SYSTEM = ['(t1 + 2*t2)*(2*t1 - t2 + 1) - 6', '(2*t1 - t2 + 1)/(t1 + 2*t2) - 2/3']  # - bad

# X0 = [0, 3]
# SYSTEM = ['t1 + t2 - 3', 't1^2 + t2^2 - 9']  # - bad

# X0 = [-0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
# X0 = [0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
# SYSTEM = ['t1^2 + t2^2 + t3^2 - 1', '2*t1^2 + t2^2 - 4*t3', '3*t1^2 - 4*t2 + t3^2']  # - bad

# X0 = [1, 2.2, 2]
# SYSTEM = ['log(t2/t3, 10) + 1 - t1', '0.4 + t3^2 - 2*t1^2 - t2', '2 + t1*t2/20 - t3']  # - bad

# X0 = [-0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
# X0 = [0.7851969330623552256, 0.496611392944656396, 0.369922830745872357]
# SYSTEM = ['t1^2 + t2^2 + t3^2 - 1', '2*t1^2 + t2^2 - 4*t3', '3*t1^2 - 4*t2 + t3^2']  # - bad

# X0 = [0.3, -0.05, -0.2, 0.3]
# SYSTEM = ['8*t1 - t2 - 2*t3 - 2.3', '10*t2 + t3 + 2*t4 + 0.5', '-t1 + 6*t3 + 2*t4 + 1.2', '3*t1 - t2 + 2*t3 + 12*t4 - 3.7']

VARIABLES = sympy.symbols('t1:' + str(len(X0) + 1))
F = sympy.sympify(np.array(SYSTEM))
N = 1000
assert N % 2 == 0 and N >= 6

if __name__ == '__main__':
    plt.gcf().canvas.mpl_connect('button_press_event', onclick)

    E = np.identity(len(X0))
    dF = sympy.Matrix(len(F), len(VARIABLES), lambda i_, j_: sympy.diff(F[i_], VARIABLES[j_]))

    r = 1
    b = [(X0_i - 1e-6, X0_i + r + 1e-6) for X0_i in X0]
    X0 = np.array(X0) + [r] * len(X0)

    # diag = np.array([lib.f_max(lambda x: lambda_from(dF[j, j])(*x), X0, b) for j in range(len(X0))])
    # if all(diag > 0):
    #     F = -F
    #     dF = -dF
    #     diag = -diag
    # elif not all(diag < 0):
    #     exit(1)
    # tmp = np.array([[lib.f_max(lambda x: abs(lambda_from(dF[k, j])(*x)), X0, b) for j in range(len(X0)) if j != k] for k in range(len(X0))])
    # non_diag = np.sum(tmp, axis=1)
    # u = np.concatenate((-diag + non_diag, -diag - non_diag))
    # diag = np.array([lib.f_min(lambda x: lambda_from(dF[j, j])(*x), X0, b) for j in range(len(X0))])
    # u = np.concatenate((u, -diag + non_diag, -diag - non_diag))

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
    print((u.max() - u.min()) / (u.max() + u.min()))


    def calc(tau, ignore=False):
        dS = E - tau * dF
        q = norm(lambda_from(dS)(*X0))
        return q if 0 <= q < 1 or ignore else np.nan


    M = norm(lambda_from(dF)(*X0))
    t = -2 * dF_X0
    non_diag_sum = np.core.umath.add.reduce(abs(t), axis=t.ndim - 1) - abs(t.diagonal())

    diag = t.diagonal()
    d = [(M - (non_diag_sum[i] + abs(diag[i] + M + (abs(diag[i] + M)))) + abs(diag[i] + M)) / 2 for i in range(len(t))]
    d = np.array(d)
    print(d)
    tau = 2 / (M + min(d))
    print('d =', min(d), 'tau =', tau)
    q = calc(2 / (M + min(d)))

    print(np.vectorize(calc)(2 / (M + d), True), 'q =', q)
    x = np.linspace(-2 * M, 2 * M, 10000)
    for g1, g2 in zip(non_diag_sum, diag):
        # print(M - g1, abs(g2 + M))
        plt.plot(x, np.vectorize(lambda x: M + (g1 + abs(g2 + M + x)))(x))
    plt.show()

    S = lambda _x: _x - tau * np.array(lambda_from(F)(*_x))
    r = norm(S(X0) - X0) / (1 - q)
    print(max(0, np.floor(np.log(0.001 / (2 * r)) / np.log(q)) + 1))
    bounds = [(X0_i - r, X0_i + r) for X0_i in X0]
    print(lib.f_max(lambda t: norm(lambda_from(dF)(*t)), X0, bounds))
    print(lib.f_min(lambda t: norm(lambda_from(dF)(*t)), X0, bounds))

    h = 2 / lib.f_max(lambda t: norm(lambda_from(dF)(*t)), X0, bounds)
    # d2F = sympy.Matrix(len(dF), len(VARIABLES), lambda i_, j_: sympy.diff(dF[i_], VARIABLES[j_]))
    # M = lib.f_max(lambda t: norm(lambda_from(dF)(*t)), X0)
    # m = lib.f_min(lambda t: norm(lambda_from(dF)(*t)), X0)
    # print(2/(m+M), m, M, la.norm(lambda_from(dF)(*X0), ord=np.inf), la.norm(lambda_from(dF)(*X0), ord=-np.inf))
    # h = 2 / norm(lambda_from(dF)(*X0))

    print(h)

    a = b = None
    i = 0
    for t in super_space(20):
        if not np.isnan(calc(h * t / 19)):
            a = 0
            b = h
            break
        elif not np.isnan(calc(-h * t / 19)):
            a = -h
            b = 0
            break
        i += 1
        if i > 1000:
            raise Exception()
    if a is None:
        raise Exception()

    ul = a
    while True:
        vl = calc(ul)
        if not np.isnan(vl):
            break
        ul += EPS

    ur = b
    while True:
        vr = calc(ur)
        if not np.isnan(vr):
            break
        ur -= EPS

    xl = yl = xr = yr = None
    kl_min = kr_min = np.inf

    points = []
    for x in np.linspace(a, b, 100)[1:-1]:
        y = calc(x)
        if not np.isnan(y):
            points.append(Point(x, y))
            kl = (y - vl) / (x - ul)
            kr = (y - vr) / (ur - x)
            if kl < kl_min:
                kl_min = kl
                xl = x
                yl = y
            if kr < kr_min:
                kr_min = kr
                xr = x
                yr = y

    left_base = [Point(ul, vl), Point(xl, yl)]
    right_base = [Point(ur, vr), Point(xr, yr)]

    left_line = right_line = None
    if vl > yl:
        left_line = line_from(*left_base)
    if vr > yr:
        right_line = line_from(*right_base)

    if left_line is not None and right_line is not None:
        intersection = lines_intersection(left_base[0], left_base[1], right_base[0], right_base[1])
    elif left_line is not None:
        intersection = points[-1]
    elif right_line is not None:
        intersection = points[0]
    else:
        raise Exception()

    print(intersection.x, calc(intersection.x))
    # plt.scatter(intersection.x, intersection.y, s=80,color='r')

    # x, y = zip(*points)
    # plt.scatter(x,y,s=10)
    # x, y = zip(*points1)
    # plt.scatter(x,y,s=10)
    x = np.linspace(a, b, N + 1)
    y = np.vectorize(calc)(x)
    plot = plt.scatter(x, y, s=6)

    # plt.scatter(*zip(*left_base))
    # plt.scatter(*zip(*right_base))
    # x,y= zip(*points)
    # plt.scatter(x, y, s=20)
    # if left_line is not None:
    #     plt.plot(x, left_line(x))
    # if right_line is not None:
    #     plt.plot(x, right_line(x))

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

    # plt.ylim(0.825, 1.01)
    # plt.xlim(a, b)
    # print('limits:\n', plt.gca().get_xlim(), plt.gca().get_ylim())
    # plt.xlabel('τ', fontsize=14)
    # plt.ylabel('q', fontsize=14)
    plt.show()

    # with open('res', 'wb') as f:
    #     pickle.dump(res, f)

    # graph.draw_screen(res)

    # lib.star_wars()
