import numpy as np
import pandas as pd
"""
105 - IB-81, 5 в списке
max((Y - Ye)^2)
"""

np.set_printoptions(precision=3)
np.seterr(divide='ignore')


def generate(start, end, size=(8, 3)):
    return np.random.randint(start, end, size)


print('Y = a0 + a1X1 + a2X2 + a3X3', end='\n\n')
# 1
start, end = 0, 20
n_factors = 3
n_nodes = 8
X = generate(start, end, (n_nodes, n_factors))
X = np.column_stack([np.ones(n_nodes), X])
print('X:\n', X[:, 1:], end='\n\n')
print(X)
# 2
a = np.array([2, 2, 3, 4])
print('a:\n', a, end='\n\n')
Y = (X * a).sum(axis=1, keepdims=True)
print('Y:\n', Y, end='\n\n')

# 3
x0 = (X.max(axis=0) + X.min(axis=0)) / 2
print('x0:\n', x0[1:], end='\n\n')
dx = X.max(axis=0) - x0
print('dx:\n', dx[1:], end='\n\n')

signs = np.sign((X - x0))
Xn = (X - x0) / dx
Xn[np.isnan(Xn)] = 1 * signs[np.isnan(Xn)]
print('Xn:\n', Xn[:, 1:], end='\n\n')

Ye = (x0 * a).sum()
print('Ye:\n', Ye, end='\n\n')

# 4
nodes = (Y - Ye) ** 2
ind = nodes.argmax()
print(f'max((Y - Ye)^2) is {nodes[ind].item()}, ind: {ind} where Y {Y[ind].item()} & Ye {Ye.item()}')


