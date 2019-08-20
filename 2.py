import numpy as np
from functools import reduce
import operator
import itertools
import math
import matplotlib.pyplot as plt
import time

max_ = -1
res = (None, set())
t = 0
N = 8
assert N % 2 == 0 and N > 4
timer = time.time()
for x in itertools.permutations(range(1, N + 1)):
    t += 1
    if time.time() - timer >= 5:
        print(100 * t / math.factorial(N))
        timer = time.time()
    s = 0
    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            s += (len(x) - abs(x[i] - x[j])) * (j - i)
    if s > max_:
        max_ = s
        res = (max_, set())
        res[1].add(x)
        # print('NEW', res)
    elif s == max_:
        res[1].add(x)
        # print('ADD', res)
print(res)

x = list(res[1])[0]
X = []
for j in range(1, N + 1):
    r = []
    for i in range(1, N + 1):
        r.append(abs(x.index(j) - x.index(i)))
    X.append(r)
print(X)
plt.imshow(X)
plt.show()