import numpy as np

'''

Вариант 25

exp(sqrt(x)) - x * ln(1 + x) - 1 / 2 = 0

'''


def func(x):
    return np.exp(np.sqrt(x)) - x * np.log(1 + x) - 1 / 2


def func_diff(x):
    return -np.log(x + 1) + np.exp(np.sqrt(x)) / 2 / np.sqrt(x) - x / (x + 1)


A = 0
B = 10

TRUE_X = 4.81002035431984177707219181


def dichotomy_method(epsilon):
    left = A
    right = B
    middle = (left + right) / 2
    while abs(func(middle)) >= epsilon:
        if func(middle) * func(left) < 0:
            right = middle
        elif func(middle) * func(right) < 0:
            left = middle
        middle = (left + right) / 2
    return middle


def newton_method(epsilon):
    x0 = 5.9882  # (A + B) / 2
    x, x_prev = x0, x0 + 2 * epsilon
    while abs(x - x_prev) >= epsilon:
        x, x_prev = x - func(x) / func_diff(x), x

    return x


print("\nEpsilon\t\tDichotomy\t\t\tError")
epsilons = [1e-03, 1e-06, 1e-09]
for epsilon in epsilons:
    x = dichotomy_method(epsilon)
    if len(str(x)) < 13:
        print("{0}\t\t{1}\t\t{2}".format(epsilon, x, abs(func(x))))
    else:
        print("{0}\t\t{1}\t{2}".format(epsilon, x, abs(func(x))))

print("\nEpsilon\t\tNewton\t\t\t\tError")
epsilons = [1e-03, 1e-06, 1e-09]
for epsilon in epsilons:
    x = newton_method(epsilon)
    if len(str(x)) < 13:
        print("{0}\t\t{1}\t\t{2}".format(epsilon, x, abs(func(x))))
    else:
        print("{0}\t\t{1}\t{2}".format(epsilon, x, abs(func(x))))

print("\nTarget value: {0:.10f}".format(TRUE_X))
