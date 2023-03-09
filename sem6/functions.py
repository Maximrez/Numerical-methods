import math
import numpy as np


def simple_mu(t):
    return 0


def phi1(x, x0=1.5, eps=0.5):
    if x0 - eps <= x <= x0 + eps:
        return 1
    return 0


def phi2(x, x0=1.5, eps=0.5):
    return phi1(x, x0, eps) * (1 - (x - x0) ** 2 / eps ** 2)


def phi3(x, x0=1.5, eps=0.5):
    if x0 - eps < x < x0 + eps:
        return math.exp(-(x - x0) ** 2 / (eps ** 2 - (x - x0) ** 2))
    return 0


def phi4(x, x0=1.5, eps=0.5):
    return phi1(x, x0, eps) * math.pow(math.cos(math.pi * abs(x - x0) / 2 / eps), 3)


def exact(phi, mu, n, T, c=0.7, l=10, a=1):
    h = l / (n - 1)
    N = int(T / (c * h / a))
    grid = [h * i for i in range(n)]
    exact_matrix = np.zeros((N, n), dtype=float)
    exact_matrix[0] = np.array(list(map(phi, grid)))

    for i in range(1, N):
        for j in range(n):
            t = i * c * h / a
            if grid[j] < a * t:
                exact_matrix[i][j] = mu(t - grid[j] / a)
            else:
                exact_matrix[i][j] = phi(grid[j] - a * t)

    return exact_matrix


def angle(phi, mu, n, T, c=0.7, l=10, a=1):
    h = l / (n - 1)
    N = int(T / (c * h / a))
    grid = [h * i for i in range(n)]
    matrix = np.zeros((N, n), dtype=float)

    u = np.array([0] + list(map(phi, grid)))
    matrix[0] = u[1:]
    for i in range(1, N):
        u[1:] = u[1:] - c * (u[1:] - u[:-1])
        u[0] = mu(c * h * i / a)
        matrix[i] = u[1:]

    return grid, matrix


def square(phi, mu, n, T, c=0.7, l=10, a=1):
    h = l / (n - 1)
    N = int(T / (c * h / a))
    grid = [h * i for i in range(n)]
    matrix = np.zeros((N, n), dtype=float)

    u = np.array(list(map(phi, grid)))
    matrix[0] = u
    for i in range(1, N):
        v = np.zeros(n, dtype=float)
        v[0] = mu(c * h * i / a)
        for j in range(1, n):
            v[j] = (-v[j - 1] + u[j - 1] + u[j] - c * (u[j] - v[j - 1] - u[j - 1])) / (1 + c)
        matrix[i] = v
        u = v

    return grid, matrix


def LAX_WN(phi, mu, n, T, c=0.7, l=10, a=1):
    h = l / (n - 1)
    N = int(T / (c * h / a))
    grid = [h * i for i in range(n)]
    mx = np.zeros((N, n), dtype=float)

    mx[0] = np.array(list(map(phi, grid)))
    for i in range(1, N):
        mx[i][0] = mu(c * h * i / a)
        for j in range(1, n - 1):
            mx[i][j] = c ** 2 / 2 * (mx[i - 1][j + 1] - 2 * mx[i - 1][j] + mx[i - 1][j - 1]) + mx[i - 1][j] - c * (mx[i - 1][j + 1] - mx[i - 1][j - 1]) / 2

    return grid, mx


def vanLeer_limiter(r):
    return (r + abs(r)) / (1 + abs(r)) / 2


def minmod_limiter(r):
    return max(0, min(1, r))


def LAXTVD(phi, mu, n, T, c=0.7, l=10, a=1):
    h = l / (n - 1)
    N = int(T / (c * h / a))

    grid = [h * i for i in range(n)]
    mx = np.zeros((N, n), dtype=float)

    mx[0] = np.array(list(map(phi, grid)))
    for i in range(1, N):
        u = np.array([mu(c * h * (i - 2) / a)] + list(mx[i - 1]) + [0])

        F_angle = np.array(u[1:n + 1])
        F_lax = 0.5 * (u[1:n + 1] * (1 + c) + u[2:n + 2] * (1 - c))

        r = np.zeros(n)
        for j in range(1, n + 1):
            if abs(u[j + 1] - u[j]) == 0 and abs(u[j] - u[j - 1]) == 0:
                r[j - 1] = 1
            elif abs(u[j + 1] - u[j]) == 0:
                r[j - 1] = 1 * np.sign(u[j] - u[j - 1])
            else:
                r[j - 1] = (u[j] - u[j - 1]) / (u[j + 1] - u[j])

        F = [F_angle[j] + minmod_limiter(r[j]) * (F_lax[j] - F_angle[j]) for j in range(n)]

        mx[i][0] = mu(c * h * i / a)
        for j in range(1, n):
            mx[i][j] = mx[i - 1][j] - c * (F[j] - F[j - 1])

    return grid, mx
