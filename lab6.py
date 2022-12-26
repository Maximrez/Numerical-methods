import numpy as np
import matplotlib.pyplot as plt

'''

Вариант 25

u" + 3(th2x)u' + (1 - th2x)u = 1/2 * (3cosx - sinx)th2x

u'(0) = 3/2
3u(1) - u'(1) = 5.1460
u0(x) = sqrt(1 + th2x) + 1/2 * sinx

'''


def u0(x):
    return np.sqrt(1 + np.tanh(2 * x)) + np.sin(x) / 2


'''
Решение:

Представляем уравнения в виде:
u" + p(x)u' + q(x)u = f(x)
a1 * u(0) + b1 * u'(0) = g1
a2 * u(1) + b2 * u'(1) = g2

'''


def p(x):
    return 3 * np.tanh(2 * x)


def q(x):
    return 1 - np.tanh(2 * x)


def f(x):
    return (3 * np.cos(x) - np.sin(x)) * np.tanh(2 * x) / 2


def equation(x, u, u1):
    return -p(x) * u1 - q(x) * u + f(x)


a1, b1, g1 = 0, 1, 3 / 2
a2, b2, g2 = 3, -1, 5.1460

h = 0.05
A, B = 0, 1

x_values = np.arange(A, B + h, h)
true_u_values = list(u0(x_values))


def progonka(a, b, c, f):
    a_values = [-c[0] / b[0]]
    b_values = [f[0] / b[0]]

    for i in range(1, len(a) - 1):
        a_values.append(-c[i] / (b[i] + a[i] * a_values[i - 1]))
    for i in range(1, len(a) - 1):
        b_values.append((f[i] - a[i] * b_values[i - 1]) / (b[i] + a[i] * a_values[i - 1]))

    a_values.append(0)
    b_values.append((f[-1] - a[-1] * b_values[-1]) / (b[-1] + a[-1] * a_values[-2]))

    # обратный ход
    y = [0] * len(a)
    y[-1] = b_values[-1]
    for i in range(len(a) - 2, -1, -1):
        y[i] = b_values[i] + a_values[i] * y[i + 1]

    return y


def difference_approach_1(x_values):
    a_values = [0]
    b_values = [a1 - b1 / h]
    c_values = [b1 / h]
    d_values = [g1]

    for i in range(1, len(x_values) - 1):
        a_values.append(1 / (h ** 2) - p(x_values[i]) / (2 * h))
    for i in range(1, len(x_values) - 1):
        b_values.append(-2 / (h ** 2) + q(x_values[i]))
    for i in range(1, len(x_values) - 1):
        c_values.append(1 / (h ** 2) + p(x_values[i]) / (2 * h))
    for i in range(1, len(x_values) - 1):
        d_values.append(f(x_values[i]))

    a_values.append(- b2 / h)
    b_values.append(a2 + b2 / h)
    c_values.append(0)
    d_values.append(g2)

    return progonka(a_values, b_values, c_values, d_values)


def difference_approach_2(x_values):
    a_values = [0]
    b_values = [-2 + (2 * a1 * h / b1) - (p(x_values[0]) * a1 * (h ** 2) / b1) + q(x_values[0]) * (h ** 2)]
    c_values = [2]
    d_values = [f(x_values[0]) * (h ** 2) + ((g1 * 2 * h) / b1) - (p(x_values[0]) * g1 * (h ** 2) / b1)]

    for i in range(1, len(x_values) - 1):
        a_values.append(1 / (h ** 2) - p(x_values[i]) / (2 * h))
    for i in range(1, len(x_values) - 1):
        b_values.append(-2 / (h ** 2) + q(x_values[i]))
    for i in range(1, len(x_values) - 1):
        c_values.append(1 / (h ** 2) + p(x_values[i]) / (2 * h))
    for i in range(1, len(x_values) - 1):
        d_values.append(f(x_values[i]))

    a_values.append(2)
    b_values.append(-2 - (2 * h * a2 / b2) - (p(x_values[-1]) * (h ** 2) * a2 / b2) + (q(x_values[-1]) * (h ** 2)))
    c_values.append(0)
    d_values.append(f(x_values[-1]) * (h ** 2) - ((h ** 2) * p(x_values[-1]) * g2 / b2) - (2 * h * g2 / b2))

    return progonka(a_values, b_values, c_values, d_values)


d1_u_values = difference_approach_1(x_values)
d2_u_values = difference_approach_2(x_values)
# print(true_u_values)
# print(d1_u_values)
# print(d2_u_values)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['font.size'] = '12'

plt.plot(x_values, d1_u_values, label='Разностный подход 1 порядка', linewidth=2)
plt.plot(x_values, d2_u_values, label='Разностный подход 2 порядка', linewidth=3)
plt.plot(x_values, true_u_values, label='true', linewidth=2)

plt.xlabel('x')
plt.ylabel('u')
plt.legend(loc='lower right')
plt.grid()
plt.show()
