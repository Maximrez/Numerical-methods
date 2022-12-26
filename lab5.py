import numpy as np
import matplotlib.pyplot as plt

'''

Вариант 25

u" + 3(th2x)u' + (1 - th2x)u = 1/2 * (3cosx - sinx)th2x

u(0) = 1
u'(0) = 3/2
u0(x) = sqrt(1 + th2x) + 1/2 * sinx

'''

'''
Решение:

Сделаем замену w = u', решим систему ОДУ 1-го порядка:
du/dx = w
dw/dx = -3(th2x)w + (th2x - 1)u + 1/2 * (3cosx - sinx)th2x

'''


def f(x, u, w):
    return -3 * np.tanh(2 * x) * w + (np.tanh(2 * x) - 1) * u + (3 * np.cos(x) - np.sin(x)) * np.tanh(2 * x) / 2


def u0(x):
    return np.sqrt(1 + np.tanh(2 * x)) + np.sin(x) / 2


h = 0.05
A, B = 0, 1
u0_value = 1.0
u10_value = 3 / 2

x_values = np.arange(A, B + h, h)
true_u_values = list(u0(x_values))


def Euler(x_values):
    u_values = [u0_value]
    w_values = [u10_value]
    i = 0
    while x_values[i] < B:
        u_values.append(u_values[i] + h * w_values[i])
        w_values.append(w_values[i] + h * f(x_values[i], u_values[i], w_values[i]))
        i += 1
    return u_values


def Runge_Kutta_4(x_values):
    u_values = [u0_value]
    w_values = [u10_value]
    i = 0
    while x_values[i] < B:
        k1 = f(x_values[i], u_values[i], w_values[i])
        q1 = w_values[i]

        k2 = f(x_values[i] + h / 2, u_values[i] + h / 2 * q1, w_values[i] + h / 2 * 1)
        q2 = w_values[i] + h / 2 * k1

        k3 = f(x_values[i] + h / 2, u_values[i] + h / 2 * q2, w_values[i] + h / 2 * k2)
        q3 = w_values[i] + h / 2 * k2

        k4 = f(x_values[i] + h, u_values[i] + h * q3, w_values[i] + h * k3)
        q4 = w_values[i] + h * k3

        w_values.append(w_values[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
        u_values.append(u_values[i] + (h / 6) * (q1 + 2 * q2 + 2 * q3 + q4))
        i += 1

    return u_values, w_values


def Adams_3(x_values, u_values, w_values):
    i = 2
    while x_values[i] < B:
        k1 = f(x_values[i], u_values[i], w_values[i])
        q1 = w_values[i]

        k2 = f(x_values[i - 1], u_values[i - 1], w_values[i - 1])
        q2 = w_values[i - 1]

        k3 = f(x_values[i - 2], u_values[i - 2], w_values[i - 2])
        q3 = w_values[i - 2]

        w_values.append(w_values[i] + h * (23 / 12 * k1 - 4 / 3 * k2 + 5 / 12 * k3))
        u_values.append(u_values[i] + h * (23 / 12 * q1 - 4 / 3 * q2 + 5 / 12 * q3))
        i += 1

    return u_values


euler_u_values = Euler(x_values)
runge_kutta_u_values, runge_kutta_u1_values = Runge_Kutta_4(x_values)
adams_u_values = Adams_3(x_values, runge_kutta_u_values[:3], runge_kutta_u1_values[:3])
# print(true_u_values)
# print(euler_u_values)
# print(runge_kutta_u_values)
# print(adams_u_values)

plt.rcParams["figure.figsize"] = (10, 8)
plt.rcParams['font.size'] = '12'

plt.plot(x_values, euler_u_values, label='Euler', linewidth=3)
plt.plot(x_values, runge_kutta_u_values, label='Runge-Kutta', linewidth=3)
plt.plot(x_values, adams_u_values, label='Adams', linewidth=4)
plt.plot(x_values, true_u_values, label='true', linewidth=3)

plt.xlabel('x')
plt.ylabel('u')
plt.title('Решения резными методами')
plt.legend(loc='lower right')
plt.grid()
plt.show()
