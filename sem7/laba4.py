import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as ssl
import scipy.sparse as ss
from matplotlib.animation import ArtistAnimation

nx = 100
ny = 100

Lx = 1
Ly = 1

hx = Lx / (nx - 1)
hy = Ly / (ny - 1)


def solution(x, y):
    return np.sin(2.7 * x) * np.cos(2.3 * y + 0.2) * 50


def phi1(y):
    return solution(0, y)


def phi2(y):
    return solution(1, y)


def psi1(x):
    return solution(x, 0)


def psi2(x):
    return solution(x, 1)


def f(x, y):
    return - np.sin(2.7 * x) * np.cos(2.3 * y + 0.2) * 50 * (2.7 ** 2) - np.sin(2.7 * x) * np.cos(2.3 * y + 0.2) * 50 * (2.3 ** 2)


def u0(x, y):
    return f(x, y)


def set_borders(u):
    for x in range(u.shape[0]):
        u[x, 0] = psi1(x * hx)
        u[x, -1] = psi2(x * hx)
    for y in range(u.shape[1]):
        u[0, y] = phi1(y * hy)
        u[-1, y] = phi2(y * hy)


def get_line(u_prev, u_next, xi, tau1, tau2):
    b = [[0] for _ in range(u_prev.shape[1])]
    rows = [0, u_next.shape[1] - 1]
    cols = [0, u_next.shape[1] - 1]
    data = [1, 1]
    b[0][0] = psi1(xi * hx)
    b[-1][0] = psi2(xi * hx)
    for i in range(1, u_prev.shape[1] - 1):
        rows.append(i)
        cols.append(i - 1)
        data.append(-tau2 / (hy ** 2))
        rows.append(i)
        cols.append(i)
        data.append(2 * tau2 / (hy ** 2) + 1)
        rows.append(i)
        cols.append(i + 1)
        data.append(-tau2 / (hy ** 2))
        b[i][0] = ((u_prev[xi + 1, i] - 2 * u_prev[xi, i] + u_prev[xi - 1, i]) / (hx ** 2) - f(xi * hx, i * hy)) * tau2 + u_prev[xi, i]
    A = ss.csr_matrix((data, (rows, cols)), u_next.shape)
    B = ss.csr_matrix(b, shape=(u_next.shape[1], 1))
    return ssl.spsolve(A, B)


def get_column(u_prev, u_next, yi, tau1, tau2):
    rows = [0, u_next.shape[0] - 1]
    cols = [0, u_next.shape[0] - 1]
    data = [1, 1]
    b = [[0] for _ in range(u_prev.shape[0])]
    b[0][0] = phi1(yi * hy)
    b[-1][0] = phi2(yi * hy)
    for i in range(1, u_prev.shape[0] - 1):
        rows.append(i)
        cols.append(i - 1)
        data.append(-tau1 / (hx ** 2))
        rows.append(i)
        cols.append(i)
        data.append(2 * tau1 / (hx ** 2) + 1)
        rows.append(i)
        cols.append(i + 1)
        data.append(-tau1 / (hx ** 2))
        b[i][0] = ((u_prev[i, yi + 1] - 2 * u_prev[i, yi] + u_prev[i, yi - 1]) / (hy ** 2) - f(i * hx, yi * hy)) * tau1 + u_prev[i, yi]
    A = ss.csr_matrix((data, (rows, cols)), u_next.shape)
    B = ss.csr_matrix(b, shape=(u_next.shape[0], 1))
    return ssl.spsolve(A, B)


def get_iteration_number(tau1=1, tau2=1, epsilon=0.1):
    u = np.asmatrix([[u0(x * hx, y * hy) for y in range(ny)] for x in range(nx)])

    # for animation
    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(projection='3d')
    x = np.linspace(0, Lx, nx, endpoint=True)
    y = np.linspace(0, Ly, ny, endpoint=True)
    xgrid, ygrid = np.meshgrid(x, y)
    frames = []

    def get_error(u):
        max_value = None
        for x in range(1, u.shape[0] - 1):
            for y in range(1, u.shape[1] - 1):
                value = (u[x + 1, y] - 2 * u[x, y] + u[x - 1, y]) / (hx ** 2) + (u[x, y + 1] - 2 * u[x, y] + u[x, y - 1]) / (hy ** 2)
                if max_value is None or max_value < abs(value - f(x * hx, y * hy)):
                    max_value = abs(value - f(x * hx, y * hy))
        for x in range(u.shape[0]):
            value = u[x, 0]
            if max_value < abs(value - psi1(x * hx)):
                max_value = abs(value - psi1(x * hx))
            value = u[x, -1]
            if max_value < abs(value - psi2(x * hx)):
                max_value = abs(value - psi2(x * hx))
        for y in range(u.shape[1]):
            value = u[0, y]
            if max_value < abs(value - phi1(y * hy)):
                max_value = abs(value - phi1(y * hy))
            value = u[-1, y]
            if max_value < abs(value - phi2(y * hy)):
                max_value = abs(value - phi2(y * hy))
        return max_value

    def get_error_from_sol(u):
        max_value = None
        for x in range(u.shape[0]):
            for y in range(u.shape[1]):
                value = u[x, y]
                if max_value is None or max_value < abs(value - solution(x * hx, y * hy)):
                    max_value = abs(value - solution(x * hx, y * hy))
        return max_value

    iteration = 0
    error = get_error_from_sol(u)
    while error >= epsilon:
        print(f'iteration: {iteration}; error: {error}')
        iteration += 1

        # u (tau + 1/2)
        u_1 = u.copy()
        set_borders(u_1)
        for i in range(1, u.shape[0] - 1):
            column = get_column(u, u_1, i, tau1, tau2)
            u_1[:, i] = column.reshape(column.shape[0], 1)

        # u (tau + 1)
        u_2 = u_1.copy()
        set_borders(u_2)
        for i in range(1, u.shape[0] - 1):
            line = get_line(u_1, u_2, i, tau1, tau2)
            u_2[i, :] = line

        error = get_error_from_sol(u_2)
        u = u_2

        line = ax_3d.plot_surface(xgrid, ygrid, u_1, cmap="rainbow")
        frames.append([line])
    print('\nResult error and plot:')
    print(f'iteration: {iteration}; error: {get_error_from_sol(u)}')
    ax_3d.plot_surface(xgrid, ygrid, u, cmap="rainbow")
    animation = ArtistAnimation(
        fig,
        frames,
        interval=100,
        blit=False,
        repeat=True
    )
    plt.show()
    animation.save("Animation4_2.gif")


get_iteration_number(2, 2, 0.5)
