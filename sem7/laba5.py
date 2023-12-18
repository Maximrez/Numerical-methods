import matplotlib.pyplot as plt
import numpy as np

nx = 100
ny = 100

Lx = 1
Ly = 1

hx = Lx / (nx - 1)
hy = Ly / (ny - 1)


def solution(x, y):
    return np.sin(2.7 * x) * np.cos(2.3 * y + 0.2) * 50


def phi1(y):
    return -np.cos(2.7 * 0) * 2.7 * np.cos(2.3 * y + 0.2) * 50


def phi2(y):
    return np.cos(2.7 * Lx) * 2.7 * np.cos(2.3 * y + 0.2) * 50


def psi1(x):
    return -np.sin(2.7 * x) * (-np.sin(2.3 * 0 + 0.2)) * 2.3 * 50


def psi2(x):
    return np.sin(2.7 * x) * (-np.sin(2.3 * Ly + 0.2)) * 2.3 * 50


def f(x, y):
    return - np.sin(2.7 * x) * np.cos(2.3 * y + 0.2) * 50 * (2.7 ** 2) - np.sin(2.7 * x) * np.cos(2.3 * y + 0.2) * 50 * (2.3 ** 2)


def u0(x, y):
    return 0


sol = np.asmatrix([[solution(x * hx, y * hy) for y in range(ny)] for x in range(nx)])
result = np.ones(shape=(nx + 2, ny + 2))
u_extent = np.zeros(shape=(nx + 2, ny + 2))
Mpsi1 = np.asmatrix([psi1(x * hx) for x in range(nx)]).reshape((nx, 1))
Mpsi2 = np.asmatrix([psi2(x * hx) for x in range(nx)]).reshape((nx, 1))
Mphi1 = np.asmatrix([phi1(y * hy) for y in range(ny)])
Mphi2 = np.asmatrix([phi2(y * hy) for y in range(ny)])

round = lambda x: 0 if abs(x) < 1e-10 else x


def A(u, r=False):
    u_extent[1:-1, 1:-1] = u[:, :]
    u_extent[[0], 1:-1] = Mphi1 * hx * (not r) + u[0, :]
    u_extent[[-1], 1:-1] = Mphi2 * hx * (not r) + u[-1, :]
    u_extent[1:-1, [0]] = Mpsi1 * hy * (not r) + u[:, 0]
    u_extent[1:-1, [-1]] = Mpsi2 * hy * (not r) + u[:, -1]
    result = ((u_extent[0:-2, 1:-1] - 2 * u_extent[1:-1, 1:-1] + u_extent[2:, 1:-1]) / (hx ** 2) +
              (u_extent[1:-1, 0:-2] - 2 * u_extent[1:-1, 1:-1] + u_extent[1:-1, 2:]) / (hy ** 2))
    return np.asmatrix([[round(result[i, j]) for j in range(result.shape[1])] for i in range(result.shape[0])])


def get_f():
    f_values = np.asmatrix([[round(f(x * hx, y * hy)) for y in range(ny)] for x in range(nx)])
    return f_values


f_values = get_f()


def get_tau(r):
    Ar = A(r, r=True)
    Ar_r = np.sum(np.asmatrix([[Ar[i, j] * r[i, j] for j in range(r.shape[1])] for i in range(r.shape[0])]))
    Ar_Ar = np.sum(np.asmatrix([[Ar[i, j] * Ar[i, j] for j in range(r.shape[1])] for i in range(r.shape[0])]))
    return Ar_r / Ar_Ar


def print_f(text, clear=False):
    print(text)
    if clear:
        mode = 'w'
    else:
        mode = 'a'
        text = '\n' + text
    with open('output5.txt', mode, encoding='utf-8') as f:
        f.write(text)


def get_iterations_number(epsilon=0.1):
    u = np.asmatrix([[u0(x * hx, y * hy) for y in range(ny)] for x in range(nx)])
    tau = 1
    iteration = 0

    error = epsilon * 2
    norm_r = epsilon * 2

    print_f('', clear=True)

    while norm_r >= epsilon:
        print_f(f'iteration: {iteration}, tau: {tau}, error: {error}, ||r||: {norm_r}')
        iteration += 1
        r = A(u) - f_values
        r -= np.mean(r)
        tau = get_tau(r)
        u = u - tau * r

        norm_r_new = np.sum(np.asmatrix([[r[i, j] * r[i, j] for j in range(r.shape[1])] for i in range(r.shape[0])])) ** 0.5
        norm_r = norm_r_new

        c = np.mean(u - sol)
        error = np.max(np.abs(np.asmatrix([[u[i, j] - sol[i, j] - c for j in range(r.shape[1])] for i in range(r.shape[0])])))

    print_f(f'iteration: {iteration}, tau: {tau}, error: {error}, ||r||: {norm_r}')
    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(projection='3d')
    x = np.linspace(0, Lx, nx, endpoint=True)
    y = np.linspace(0, Ly, ny, endpoint=True)
    xgrid, ygrid = np.meshgrid(x, y)
    ax_3d.plot_surface(xgrid, ygrid, u, cmap="rainbow")
    plt.show()


get_iterations_number(0.5)
