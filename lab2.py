import numpy as np
from sympy import *
from matplotlib import pyplot as plt
import random

'''

Вариант 25

f(x)=sin(x)/(2+ln(1+x^2))

'''

A = -1
B = 1


def func(x):
    return np.sin(x) / (2 + np.log(1 + x ** 2))


def right(curr, next, h):
    return (next - curr) / h


def right_diff(x, n, h):
    y = func(x)
    return [right(y[i], y[i + 1], h) for i in range(0, n - 1)]


def center(prev, next, h):
    return (next - prev) / 2 / h


def center_diff(x, n, h):
    y = func(x)
    return [center(y[i - 1], y[i + 1], h) for i in range(1, n - 1)]


def d2_2acc(prev, curr, next, h):
    return (prev - 2 * curr + next) / h ** 2


def diff2_2acc(x, n, h):
    y = func(x)
    return [d2_2acc(y[i - 1], y[i], y[i + 1], h) for i in range(1, n - 1)]


def d2_4acc(prevprev, prev, curr, next, nextnext, h):
    return (-prevprev + 16 * prev - 30 * curr + 16 * next - nextnext) / 12 / h ** 2


def diff2_4acc(x, n, h):
    y = func(x)
    return [d2_4acc(y[i - 2], y[i - 1], y[i], y[i + 1], y[i + 2], h) for i in range(2, n - 2)]


def difference(result, true):
    return max(abs(true[i] - result[i]) for i in range(len(result)))


def calc_tan(x, y):
    num = 50
    tan = 0
    for i in range(num):
        first_idx = random.randint(0, len(x) - 1)
        second_idx = random.randint(0, len(x) - 1)
        while second_idx == first_idx:
            second_idx = random.randint(0, len(x) - 1)
        tan += (log(y[first_idx]) - log(y[second_idx])) / (log(x[first_idx]) - log(x[second_idx]))
    return tan / num


def make_first_derivative(n, show=False):
    points = np.linspace(A, B, n)
    h = (B - A) / (n - 1)

    var('x f d_f d2_f')
    f = sin(x) / (2 + log(1 + x ** 2))
    d_f = simplify(diff(f, x))

    d_f_values = [d_f.subs({'x': p}).evalf() for p in points]

    right_diff_values = right_diff(points, n, h)
    center_diff_values = center_diff(points, n, h)

    if show:
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.rcParams['font.size'] = '12'

        plt.plot(points, d_f_values, label='Function', linewidth=3)
        plt.plot(points[:-1], right_diff_values, label='Right', linewidth=3)
        plt.plot(points[1:-1], center_diff_values, label='Center', linewidth=3)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('First diff')
        plt.legend()
        plt.grid()
        plt.show()

    return difference(right_diff_values, d_f_values[:-1]), difference(center_diff_values, d_f_values[1:-1]), h


def make_second_derivative(n, show=False):
    points = np.linspace(A, B, n)
    h = (B - A) / (n - 1)

    var('x f d_f d2_f')
    f = sin(x) / (2 + log(1 + x ** 2))
    d2_f = simplify(diff(diff(f, x), x))

    d2_f_values = [d2_f.subs({'x': p}).evalf() for p in points]

    diff2_2acc_values = diff2_2acc(points, n, h)
    diff2_4acc_values = diff2_4acc(points, n, h)

    if show:
        plt.rcParams["figure.figsize"] = (10, 8)
        plt.rcParams['font.size'] = '12'

        plt.plot(points, d2_f_values, label='Function', linewidth=3)
        plt.plot(points[1:-1], diff2_2acc_values, label='2 acc', linewidth=3)
        plt.plot(points[2:-2], diff2_4acc_values, label='4 acc', linewidth=3)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Second diff')
        plt.legend()
        plt.grid()
        plt.show()

    return difference(diff2_2acc_values, d2_f_values[1:-1]), difference(diff2_4acc_values, d2_f_values[2:-2]), h


print("Первая производная:\nmax|right-true| = {:.7f}\nmax|center-true| = {:.7f}\nh = {:.4f}".format(*make_first_derivative(50, True)))
print()
print("Вторая производная:\nmax|2acc-true| = {:.7f}\nmax|4acc-true| = {:.7f}\nh = {:.4f}".format(*make_second_derivative(50, True)))

h_values = []
right_errors = []
center_errors = []
acc2_errors = []
acc4_errors = []
for n in range(10, 101, 10):
    right_error, center_error, h_value = make_first_derivative(n)
    acc2_error, acc4_error, _ = make_second_derivative(n)

    h_values.append(h_value)
    right_errors.append(right_error)
    center_errors.append(center_error)
    acc2_errors.append(acc2_error)
    acc4_errors.append(acc4_error)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['font.size'] = '12'

plt.loglog(h_values, right_errors, label='right ({:.2f})'.format(calc_tan(h_values, right_errors)), linewidth=3)
plt.loglog(h_values, center_errors, label='center ({:.2f})'.format(calc_tan(h_values, center_errors)), linewidth=3)
plt.loglog(h_values, acc2_errors, label='2 acc ({:.2f})'.format(calc_tan(h_values, acc2_errors)), linewidth=3)
plt.loglog(h_values, acc4_errors, label='4 acc ({:.2f})'.format(calc_tan(h_values, acc4_errors)), linewidth=3)

plt.xlabel('log(h)')
plt.ylabel('log(max|error|)')
plt.title('Зависимость логарифма максимальной ошибки от логарифма шага')
plt.legend(loc='lower right')
plt.grid()
plt.show()
