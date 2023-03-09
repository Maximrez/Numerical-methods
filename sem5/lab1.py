from numpy import linspace
import math
from matplotlib import pyplot as plt
from sympy import *
import time

n = 20
a = 0
b = 10


# Вариант 25
def func(x):
    # return math.cos(x / 4) * (math.sin(x / 2)) ** 2
    # return math.tanh(x ** 3) / (x + 1)
    # return math.exp(x / 3) / (1 + x ** 2)
    return math.sin(3 * x) * x


def interpolate(points, values, name):
    var('x phi f')
    f = 0
    for i in range(n):
        phi = values[i]
        for k in range(n):
            if k != i:
                phi *= (x - points[k]) / (points[i] - points[k])
        f += phi
    f = simplify(expand(f))

    m = 200  # Количество точек для построения графика
    x_range = linspace(a, b, m)

    mse = 0
    mae = 0
    maxe = 0
    y_polinom = list(f.subs({x: i}) for i in x_range)
    y_func = list(func(i) for i in x_range)
    for i in range(m):
        diff = abs(y_func[i] - y_polinom[i])
        mse += diff ** 2
        mae += diff
        maxe = max(diff, maxe)

    mse = math.sqrt(mse / m / (m - 1))
    mae /= m

    plt.rcParams["figure.figsize"] = (10, 8)
    plt.rcParams['font.size'] = '12'

    plt.plot(x_range, y_func, label='Function', linewidth=3)
    plt.plot(x_range, y_polinom, label='Polinom', linewidth=3)
    plt.scatter(x=points, y=values, label='Train points', s=50, c='g')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(name)
    plt.legend()
    plt.grid()
    plt.show()

    print("MSE: %.10f, MAE: %.10f, Max error: %.10f" % (mse, mae, maxe))


def main(points, name):
    print("\n" + name + ":")
    start = time.time()
    values = list(func(i) for i in points)
    interpolate(points, values, name)
    end = time.time()
    print('Time: %.5f seconds' % float(end - start))


if __name__ == '__main__':
    points = linspace(a, b, n)
    main(points, 'Фиксированный шаг')

    points = list((a + b) / 2 + (b - a) / 2 * math.cos((2 * k - 1) * math.pi / (2 * n)) for k in range(1, n + 1))
    main(points, 'Узлы Чебышева')
