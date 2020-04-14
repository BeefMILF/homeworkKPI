import numpy as np


np.set_printoptions(precision=4, suppress=True)
"""
105 
    X1          X2           X3
min   max   min    max   min     max
-30   20    15     50    20     35
"""

# 1
print('------------     Task 1      ------------')
print('b0 + b1x1 + b2x2 + b3x3\n')


# 2
print('------------     Task 2      ------------')
X = np.array([
    [1, -1, -1, -1],
    [1, -1, +1, +1],
    [1, +1, -1, +1],
    [1, +1, +1, -1]
])
print('X: ', X, sep='\n', end='\n\n')


# 3
print('------------     Task 3      ------------')
Xn = np.array([
    [-30, 15, 20],
    [-30, 50, 35],
    [20, 15, 35],
    [20, 50, 20]
])
print('Xn: ', Xn, sep='\n', end='\n\n')

ymin, ymax = 200 + np.mean([-30, 15, 20]), 200 + np.mean([20, 50, 35])
yn = np.random.randint(ymin, ymax, size=(4, 3))
print(f'ymin: {ymin:.3f}, ymax: {ymax:.3f}')
print('yn: ', yn, sep='\n', end='\n\n')


# 4
print('------------     Task 4      ------------')
y_ = yn.mean(axis=1)
print('y_: ', y_, sep='\n', end='\n\n')

Xm = Xn.mean(axis=0)
print('Xm: ', Xm, sep='\n', end='\n\n')

ym = y_.mean()
print(f'ym: {ym}\n')
print(Xn.T.shape, y_.shape)
an = np.mean(Xn.T * y_, axis=1)
print('an: ', an, sep='\n', end='\n\n')

ann = np.array([(Xn.T * Xn[:, i]).mean(axis=1) for i in range(Xn.shape[1])])
print('ann: ', ann, sep='\n', end='\n\n')

M = np.row_stack([np.hstack([1, Xm]), np.hstack([Xm.reshape(-1, 1), ann])])
A = np.linalg.det(M)
An = []
for i in range(M.shape[1]):
    M_ = np.copy(M)
    M_[:, i] = np.hstack([ym, an])
    An.append(np.linalg.det(M_))

b = np.array(An) / A
print('b: ', b, sep='\n', end='\n\n')

reg = np.insert(Xn, 0, 1, axis=1).dot(b)

print(f'b0 + b1x1 + b2x2 + b3x3')
print('\n'.join([f'y{j} = '+ ' + '.join([f'{xn[i]:.3f} * {b[i]:.3f}' for i in range(len(xn))]) + f' = {reg[j]:.3f}' for j, xn in enumerate(np.insert(Xn, 0, 1, axis=1))]))

if np.allclose(reg, y_):
    print('Перевірка правильна\n(підставимо значення факторів з матриці планування і порівняємо результат з середніми значеннями функції відгуку за рядками)\n')


# 5
print('------------     Task 5      ------------')

def ifUniformVar(x, m=3, N=4, p=0.05):
    """ Перевірка однорідності дисперсії за критерієм Кохрена """
    xm = x.mean(axis=1, keepdims=True)
    var = ((x - xm) ** 2).mean(axis=1)
    Gt = 0.7679  # таблиця
    G = var.max() / var.sum()
    if G < Gt:
        print(f'G < Gt, {G:.3f} < {Gt:.3f} - Дисперсія однорідна.')
        return True
    return False


def ifSignificant(x, y, m=3, N=4):
    """ Значимість коефіцієнтів регресії згідно критерію Стьюдента
        :return: nd.array - індекси значимих коефіцієнтів рівняння регресії
     """
    ym = y.mean(axis=1, keepdims=True)
    var = ((y - ym) ** 2).mean(axis=1)
    std = np.sqrt(var.mean() / (m * N))
    b = np.mean(x.T * ym.flatten(), axis=1)
    t = abs(b) / std
    f1, f2 = m - 1, N
    f3 = f1 * f2
    Tt = 2.306  # таблиця
    print('t: ', t, sep='\n', end='\n\n')
    ind = np.argwhere(t < Tt).ravel()
    print(f'{" ".join([f"t{i} = {t[i]:.3f}" for i in ind])} < Tt = {Tt}')
    print(f'{" ".join([f"b{i}" for i in ind])} коефіцієнти рівняння регресії приймаємо незначними при рівні значимості 0.05\n')

    return np.argwhere(t > Tt).ravel()


ifUniformVar(yn)
ind_sign = ifSignificant(X, yn)

print(f'b0 + b1x1 + b2x2 + b3x3')
reg = np.insert(Xn, 0, 1, axis=1)[:, ind_sign].dot(b[ind_sign])
print('\n'.join([f'y{j} = ' + ' + '.join([f'{xn[i]:.3f} * {b[i]:.3f}' for i in range(len(xn))]) + f' = {reg[j]:.3f}' for j, xn in enumerate(np.insert(Xn, 0, 1, axis=1)[:, ind_sign])]), end='\n\n')


def AppropriateReg(y, y_, yn, d, m=3, N=4):
    """

    :return:
    """
    f3 = (m - 1) * N
    f4 = N - d
    Ft = 4.5    # таблиця
    var_appropriate = (m / 1 if not (N - d) else (N - d)) * ((y - y_) ** 2).sum()
    var_b = yn.var(axis=1).sum() / N
    Fp = var_appropriate / var_b
    message = lambda x1, x2: f'Fp {x1} Ft, {Fp:.3f} {x1} {Ft:.3f}\nрівняння регресії {x2}адекватнe оригіналу при рівні значимості 0.05'
    if Fp > Ft:
        print(message('>', 'не'))
    else:
        print(message('<', ''))


AppropriateReg(reg, y_, yn, len(ind_sign))