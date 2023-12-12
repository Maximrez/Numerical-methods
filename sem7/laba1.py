import numpy as np
import matplotlib.pyplot as plt

n = 5

A1 = np.random.random((n, n)) * 2 - 1
b1 = np.random.random((n, 1)) * 2 - 1
A2 = np.copy(A1)
b2 = np.copy(b1)
Atest = np.copy(A1)
btest = np.copy(b1)

x0 = np.linalg.solve(A1, b1)


def solve_gauss(A: np.ndarray, b: np.ndarray):
    result = np.zeros(n)
    for i in range(n - 1):
        for j in range(i, n - 1):
            ksi = A[j + 1][i] / A[i][i]
            b[j + 1][0] -= ksi * b[i][0]
            A[j + 1] -= ksi * A[i]
    result[-1] = b[-1][0] / A[-1][-1]
    for i in range(n - 2, -1, -1):
        result[i] = (b[i] - np.sum(result * A[i])) / A[i][i]
    return result.reshape(n, 1)


result_gauss = solve_gauss(A1, b1)


def errors(x0: np.ndarray, result: np.ndarray):
    maxmod = np.max(np.abs(x0 - result))
    delta_result = np.max(np.abs(np.dot(Atest, result) - btest))
    delta_x0 = np.max(np.abs(np.dot(Atest, x0) - btest))
    return maxmod, delta_result, delta_x0


print(*errors(x0, result_gauss), sep='\n')

cur = 10
count = 10
step = 1.5

x = np.zeros(count)
y_maxmod = np.zeros(count)
y_delta_result = np.zeros(count)
y_delta_x0 = np.zeros(count)
for i in range(count):
    n = cur
    x[i] = cur

    A1 = np.random.random((n, n)) * 2 - 1
    b1 = np.random.random((n, 1)) * 2 - 1
    A2 = np.copy(A1)
    b2 = np.copy(b1)
    Atest = np.copy(A1)
    btest = np.copy(b1)

    x0 = np.linalg.solve(A1, b1)

    result_gauss = solve_gauss(A1, b1)

    y_maxmod[i], y_delta_result[i], y_delta_x0[i] = errors(x0, result_gauss)

    cur = int(step * cur)

plt.figure(figsize=(12, 8))
plt.loglog(x, y_maxmod)
plt.loglog(x, y_delta_result)
plt.loglog(x, y_delta_x0)
plt.grid()
plt.title('Errors')
plt.xlabel('Matrix size')
plt.ylabel('Errors')
plt.legend(['maxmod', 'delta_result', 'delta_x0'])
plt.show()
