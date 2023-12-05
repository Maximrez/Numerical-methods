import numpy as np
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt


def U(x):
    if x < 1 / 4:
        return 0
    elif x < 3 / 4:
        return 1000
    else:
        return 0


N = 1001
l = 1
h = l / N
A = np.zeros((N - 1, N - 1))
A[0][0] = U(h) + 2 / h ** 2
A[0][1] = -1 / h ** 2
A[N - 2][N - 2] = U((N - 1) * h) + 2 / h ** 2
A[N - 2][N - 3] = -1 / h ** 2
for i in range(1, N - 2):
    A[i][i] = U((i + 1) * h) + 2 / h ** 2
    A[i][i + 1] = -1 / h ** 2
    A[i][i - 1] = -1 / h ** 2


def rayleigh(A, epsilon, mu, b):
    y = ssl.spsolve(A - mu * np.identity(A.shape[0]), b)
    b = y / np.linalg.norm(y)
    mu = np.dot(np.dot(np.reshape(b, (1, N - 1)), A), b) / (np.dot(np.reshape(b, (1, N - 1)), b))
    err = np.max(np.abs(mu * b - np.dot(A, b)))

    while err > epsilon:
        y = ssl.spsolve(A - mu * np.identity(A.shape[0]), b)
        b = y / np.linalg.norm(y)
        mu = np.dot(np.dot(np.reshape(b, (1, N - 1)), A), b) / (np.dot(np.reshape(b, (1, N - 1)), b))
        err = np.max(np.abs(mu * b - np.dot(A, b)))

    return mu, b


b = np.reshape(np.ones(N - 1), (N - 1, 1))
b[N // 2:] *= -1
result = rayleigh(A, 1e-7, 0, b)
print(result[0])

x = np.linspace(0, 1, N + 1, endpoint=True)
y = np.reshape(result[1], (1, N - 1)).flatten()
y = np.insert(y, 0, 0)
y = np.append(y, 0)

plt.figure(figsize=(12, 8))
plt.plot(x, y)
plt.grid()
plt.title("СФ")
plt.xlabel("x")
plt.ylabel("СФ")
plt.show()
