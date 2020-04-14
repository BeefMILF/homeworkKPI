import numpy as np
"""
105 - IB-81, 5 в списке
"""

np.set_printoptions(precision=3)
np.seterr(divide='ignore')


def norm(x):
    """ Return normalized matrix of features [-1; +1] """
    x0 = (x.max(axis=0) + x.min(axis=0)) / 2
    dx = x.max(axis=0) - x0
    return (x - x0) / dx

def check_romanov_cr(x, m=5):
    """ Checks dispersion uniformity by Romanovsky criterion """
    xmean = x.mean(axis=1, keepdims=True)
    xvar = sorted((np.power(x - xmean, 2)).mean(axis=1), reverse=True)
    var0 = lambda m: np.sqrt(4 * (m - 1) / (m * (m - 4)))
    var0_ = var0(m)

    Fuv = []
    for i in range(len(xvar)):
        for xvar_ in xvar[i + 1:]:
            Fuv.append(xvar[i] / (xvar_ + 1e-12))
    Fuv = np.array(Fuv)

    Tuv = (m - 2) / m * Fuv
    Ruv = abs(Tuv - 1) / var0_
    return np.all(Ruv < 2), Ruv


# init
# Nv = 105
# ymin, ymax = (20 - Nv) * 10, (30 - Nv) * 10
# x1min, x1max = -25, 75
# x2min, x2max = 5, 40
# N, m = 3, 5

Nv = 105
ymin, ymax = (20 - Nv) * 10, (30 - Nv) * 10
x1min, x1max = -30, 20
x2min, x2max = 15, 50
N, m = 3, 5

# 1-2
X = np.array([
    [x1min, x2min],
    [x1max, x2min],
    [x1min, x2max]
])
Xn = norm(X)
print(f'Xn =\n{X}\n')
print(f'Xn =\n{Xn}\n')

# 3
# Y = np.array([
#     [9.0, 10.0, 11.0, 15.0, 9.0],
#     [15.0, 14.0, 10.0, 12.0, 14.0],
#     [20.0, 18.0, 12.0, 10.0, 16.0]
# ])
Y = np.random.randint(ymin, ymax, (N, m))
print(f'Y =\n{Y}\n')

# 4
var_uniform, vars = check_romanov_cr(Y, m)

print(f'{vars}')
if var_uniform:
    print('дисперсія однорідна\n')
else:
    print('дисперсія не однорідна\n')

# 5
mx = Xn.mean(axis=0)
my = Y.mean(axis=1).mean()  # Y.mean()
print(f'mx =\n{mx}')
print(f'my =\n{my}\n')

a1 = (Xn[:, 0] ** 2).mean()
a2 = (Xn[:, 0] * Xn[:, 1]).mean()
a3 = (Xn[:, 1] ** 2).mean()
print(f'a1 ={a1:.3f}')
print(f'a2 ={a2:.3f}')
print(f'a3 ={a3:.3f}\n')

ymean = Y.mean(axis=1, keepdims=False)
a11, a22 = Xn.T.dot(ymean) / Xn.shape[0]  # (Xn.T * ymean).mean(axis=1) [2, 3]x[3, ]
print(f'a11 ={a11:.3f}')
print(f'a22 ={a22:.3f}\n')

A = np.array([
    [1, mx[0], mx[1]],
    [mx[0], a1, a2],
    [mx[1], a2, a3],
])

C = np.array([
    [my],
    [a11],
    [a22],
])
b = np.linalg.solve(A, C)  # np.linalg.inv(A).dot(C)
print(f'A =\n{A}\n')
print(f'C =\n{C}\n')
print(f'b =\n{b}\n')

Xn = np.insert(Xn, 0, 1, axis=1)
if np.allclose(Xn.dot(b), Y.mean(axis=1, keepdims=True)):
    print('Результат збігається з середніми значеннями')

# 6
dx1, dx2 = abs(x1max - x1min) / 2, abs(x2max - x2min) / 2
print(f'dx1 ={dx1:.3f}')
print(f'dx2 ={dx2:.3f}\n')

x10, x20 = (x1max + x1min) / 2, (x2max + x2min) / 2
print(f'x10 ={x10:.3f}')
print(f'x20 ={x20:.3f}\n')

a0 = np.array([1, -x10 / dx1, -x20 / dx2]).dot(b).item()
a1 = b[1].item() / dx1
a2 = b[2].item() / dx2
print(f'a0 ={a0:.3f}')
print(f'a1 ={a1:.3f}')
print(f'a2 ={a2:.3f}\n')

X = np.insert(X, 0, 1, axis=1)
if np.allclose(X.dot(np.array([a0, a1, a2])), Y.mean(axis=1)):
    print('Отже, коефіцієнти натуралізованого рівняння регресії вірні')
