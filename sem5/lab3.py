import numpy as np
from sympy import *
from matplotlib import pyplot as plt
import random

'''

Вариант 25

f(x)=10x^4/(1-x^5)

'''

A = -1
B = 0.5


def func(x):
    return 10 * x ** 4 / (1 - x ** 5)


var('x f')
f = 10 * x ** 4 / (1 - x ** 5)
TARGET_VALUE = integrate(f, (x, A, B)).evalf()


def count_rectangles(n):
    h = (B - A) / (n - 1)
    points = np.linspace(A, B, n)

    sum = 0
    for i in range(n - 1):
        point = points[i] + h / 2
        sum += func(point) * h

    return sum


def count_trapezoids(n):
    h = (B - A) / (n - 1)
    points = np.linspace(A, B, n)

    sum = 0
    for i in range(n - 1):
        point = points[i]
        sum += h * (func(point) + func(point + h)) / 2

    return sum


def count_simpson(n):
    h = (B - A) / (n - 1)
    points = np.linspace(A, B, n)

    sum = 0
    for i in range(n - 1):
        point = points[i]
        sum += h * (func(point) + 4 * func(point + h / 2) + func(point + h)) / 6

    return sum


n = 500
step = (B - A) / (n - 1)

print("step: {:.3f}\ntarget: {:.10f}\nrectangles: {:.10f}\ntrapezoids: {:.10f}\nsimpson: {:.10f}".format(step, TARGET_VALUE, count_rectangles(n), count_trapezoids(n), count_simpson(n)))

rectangles_errors = list()
trapezoids_errors = list()
simpson_errors = list()

steps = np.logspace(-3, -1, 20)
for h in steps:
    n = int((B - A) / h + 1)
    rectangles_errors.append(abs(count_rectangles(n) - TARGET_VALUE))
    trapezoids_errors.append(abs(count_trapezoids(n) - TARGET_VALUE))
    simpson_errors.append(abs(count_simpson(n) - TARGET_VALUE))


def calc_tan(x, y, num=20):
    tan = 0
    for i in range(num):
        first_idx = random.randint(0, len(x) - 1)
        second_idx = random.randint(0, len(x) - 1)
        while second_idx == first_idx or ((log(y[first_idx]) - log(y[second_idx])) / (log(x[first_idx]) - log(x[second_idx]))) == np.nan:
            second_idx = random.randint(0, len(x) - 1)

        tan += (log(y[first_idx]) - log(y[second_idx])) / (log(x[first_idx]) - log(x[second_idx]))
    return tan / num


plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['font.size'] = '12'

plt.loglog(steps, rectangles_errors, label='rectangles ({:.2f})'.format(calc_tan(steps, rectangles_errors)), linewidth=3)
plt.loglog(steps, trapezoids_errors, label='trapezoids ({:.2f})'.format(calc_tan(steps, trapezoids_errors)), linewidth=3)
plt.loglog(steps, simpson_errors, label='simpson ({:.2f})'.format(calc_tan(steps, simpson_errors)), linewidth=3)

plt.xlabel('log(h)')
plt.ylabel('log|error|')
plt.title('Зависимость логарифма ошибки от логарифма шага')
plt.legend(loc='lower right')
plt.grid()
plt.show()

target_accuracy = 1e-06
print("\nПоиск шага интегрирования с заданной точностью:", target_accuracy)

n = 50
trapezoids_error = abs(count_trapezoids(n) - TARGET_VALUE)
while trapezoids_error > target_accuracy:
    n += 10
    trapezoids_error = abs(count_trapezoids(n) - TARGET_VALUE)

step = (B - A) / (n - 1)
print("Трапеции: кол-во точек - {0}, шаг - {1:.5f}, ошибка - {2}".format(n, step, trapezoids_error))

n = 5
simpson_error = abs(count_simpson(n) - TARGET_VALUE)
while simpson_error > target_accuracy:
    n += 5
    simpson_error = abs(count_simpson(n) - TARGET_VALUE)

step = (B - A) / (n - 1)
print("Симпсон: кол-во точек - {0}, шаг - {1:.5f}, ошибка - {2}".format(n, step, simpson_error))
