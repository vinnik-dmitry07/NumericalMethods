import numpy as np
from functools import reduce
import operator


def prod(arr):
    return reduce(operator.mul, arr, 1)


max_ = 5.3624166741636685
res = []
res1 = []
while True:
    x = np.random.choice(15, 15, replace=False) + 1
    s = 1
    l = len(x)
    for i in range(1, l):
        s *= float(prod(abs(x[:-i] - x[i:]).tolist()) ** (1 / (l - i)))
    r = s ** (1 / l)
    if r > max_:
        max_ = r
        res = x
        print(max_, res)


max_ = 54.55238095238094
res = []
while True:
    x = np.random.choice(15, 15, replace=False) + 1
    l = len(x)
    s = [sum(abs(x[:-i] - x[i:])) for i in range(1, l)]
    r = np.average(s, weights=np.arange(l-1, 0, -1))
    if r > max_:
        max_ = r
        res = [(max_, s, x)]
        print('NEW', max_, s, x)
    elif abs(r - max_) <= np.finfo(float).eps:
        t = (max_, s, x)
        if t not in res:
            res.append(t)
        print('ADD', res)

import numpy as np
from functools import reduce
import operator
import itertools
import math
max_ = 54.55238095238093
res = []
i = 0
N = 11
for x in itertools.permutations(range(1, N+1)):
    i+=1
    o=i/math.factorial(N)
    l = len(x)
    s = [sum(map(lambda a, b: abs(a-b), x[:-i], x[i:])) for i in range(1, l)]
    r = np.average(s, weights=np.arange(l - 1, 0, -1))
    if r > max_:
        max_ = r
        res = [(max_, s, x)]
        print('NEW', max_, s, x)
    elif abs(r - max_) <= np.finfo(float).eps:
        t = (max_, s, x)
        if t not in res:
            res.append(t)
        print('ADD', res)
