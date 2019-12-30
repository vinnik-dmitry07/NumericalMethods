import numpy
import math


def get_i():
    return math.e ** 1 - math.e ** 0


def simpson_method(func, mim_lim, max_lim, delta):
    def integrate(func, mim_lim, max_lim, n):
        integral = 0.0
        step = (max_lim - mim_lim) / n
        for x in numpy.arange(mim_lim + step / 2, max_lim - step / 2, step):
            integral += step / 6 * (func(x - step / 2) + 4 * func(x) + func(x + step / 2))
        return integral

    d, n = 1, 1
    while math.fabs(d) > delta:
        d = (integrate(func, mim_lim, max_lim, n * 2) - integrate(func, mim_lim, max_lim, n)) / 15
        n *= 2

    a = math.fabs(integrate(func, mim_lim, max_lim, n))
    b = math.fabs(integrate(func, mim_lim, max_lim, n)) + d
    if a > b:
        a, b = b, a
    print('Simpson:')
    print('\t%s\t%s\t%s' % (n, a, b))


simpson_method(lambda x: math.e ** x, 0.0, 1.0, 0.001)

print('True value:\n\t%s' % get_i())
