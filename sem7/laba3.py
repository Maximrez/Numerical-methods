import numpy as np
import scipy.sparse.linalg as ssl
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import SparseEfficiencyWarning

warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


def U(x):
    if abs(x - 0.5) < 0.25:
        return -500
    else:
        return 0


N = 1001
l = 1
h = l / N
A = np.zeros((N - 1, N - 1))
x = np.linspace(0, l, N - 1)
A[0][0] = U(x[0]) + 2 / h ** 2
A[0][1] = -1 / h ** 2
A[-1][-1] = U(x[-1]) + 2 / h ** 2
A[-1][-2] = -1 / h ** 2
for i in range(1, N - 2):
    A[i][i] = U(x[i]) + 2 / h ** 2
    A[i][i + 1] = -1 / h ** 2
    A[i][i - 1] = -1 / h ** 2


def rayleigh_method(A, b_norm, eps: float = 1e-6, mu: float = 100.0):
    diff = 1e-1
    while eps < diff:
        b_new = ssl.spsolve(A - mu * np.identity(b_norm.shape[0]), b_norm)
        b_norm = b_new / np.linalg.norm(b_new)
        mu = (A @ b_norm) @ b_norm
        diff = np.max(np.abs(A @ b_norm - mu * b_norm))

    return b_norm, mu


fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title('Eigenfunction')
ax.set_xlabel("x")
ax.set_ylabel("y")

eigenvalues = list()

# 0
b0 = np.reshape(np.ones(N - 1), (N - 1, 1))
ans = rayleigh_method(A, b0, mu=-500)
eigenvalue = ans[1]
print(eigenvalue)
eigenvalues.append(eigenvalue)
y = ans[0]
ax.plot(x, y)

# 1
b0 = np.reshape(np.ones(N - 1), (N - 1, 1))
b0[N // 2:] *= -1
ans = rayleigh_method(A, b0, mu=-1000)
eigenvalue = ans[1]
print(eigenvalue)
eigenvalues.append(eigenvalue)
y = ans[0]
ax.plot(x, y)

# 2
b0 = np.reshape(np.ones(N - 1), (N - 1, 1))
ans = rayleigh_method(A, b0, mu=-4000)
eigenvalue = ans[1]
print(eigenvalue)
eigenvalues.append(eigenvalue)
y = ans[0]
ax.plot(x, y)

# 3
b0 = np.reshape(np.ones(N - 1), (N - 1, 1))
b0[N // 2:] *= -1
ans = rayleigh_method(A, b0, mu=-4000)
eigenvalue = ans[1]
print(eigenvalue)
eigenvalues.append(eigenvalue)
y = ans[0]
ax.plot(x, y)

# 4
b0 = np.ones((N - 1)) / np.linalg.norm(np.ones((N - 1, 1)))
ans = rayleigh_method(A, b0, mu=500)
eigenvalue = ans[1]
print(eigenvalue)
eigenvalues.append(eigenvalue)
y = ans[0]
ax.plot(x, y)

# 10
# b0 = np.reshape(np.ones(N - 1), (N - 1, 1))
# ans = rayleigh_method(A, b0, mu=1000)
# eigenvalue = ans[1]
# print(eigenvalue)
# eigenvalues.append(eigenvalue)
# y = ans[0]
# ax.plot(x, y)

plt.legend(eigenvalues)
plt.grid()
plt.show()
