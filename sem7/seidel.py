import numpy as np
import matplotlib.pyplot as plt
import time

n = 5

A1 = np.random.random((n, n)) * 2 - 1 + np.identity(n) * (np.random.random((n, n)) + n + 1)
b1 = np.random.random((n, 1)) * 2 - 1
A2 = np.copy(A1)
b2 = np.copy(b1)
Atest = np.copy(A1)
btest = np.copy(b1)

x0 = np.linalg.solve(A1, b1)


def errors(x0: np.ndarray, result: np.ndarray):
    maxmod = np.max(np.abs(x0 - result))
    deltag = np.max(np.abs(np.dot(Atest, result) - btest))
    delta0 = np.max(np.abs(np.dot(Atest, x0) - btest))
    return maxmod, deltag, delta0


def iter_seidel(A: np.ndarray, b: np.ndarray, result: np.ndarray):
    result_ = np.copy(result)
    for i in range(n):
        result_[i][0] = (b[i][0] - np.sum(np.dot(A[i], result_)) + A[i][i] * result[i][0]) / A[i][i]
    return result_


# delta = 1e-14
def solve_seidel(A: np.ndarray, b: np.ndarray, error: int=0, delta=1e-14):
    result = np.zeros((n, 1))
    for i in range(n):
        result[i][0] = b[i][0] / A[i][i]

    iter_count = 0
    while (errors(x0, iter_seidel(A, b, result))[error] >= delta):
        result = iter_seidel(A, b, result)
        iter_count += 1
    return result, iter_count


# cur = 10
# count = 10
# step = 1.5
#
# x = np.zeros(count)
# y = np.zeros((3, count))
# for i in range(count):
#     n = cur
#     x[i] = cur
#
#     A1 = np.random.random((n, n)) * 2 - 1 + np.identity(n) * (np.random.random((n, n)) + n + 1)
#     b1 = np.random.random((n, 1)) * 2 - 1
#     A2 = np.copy(A1)
#     b2 = np.copy(b1)
#     Atest = np.copy(A1)
#     btest = np.copy(b1)
#
#     x0 = np.linalg.solve(A1, b1)
#     for j in range(3):
#         start = time.time()
#         solve_seidel(A2, b2, j)
#         end = time.time()
#         y[j][i] = (end - start) * 10 ** 3
#
#     cur = int(step * cur)

# plt.figure(figsize=(12, 8))
# plt.loglog(x, y[0])
# plt.loglog(x, y[1])
# plt.loglog(x, y[2])
# plt.grid()
# plt.title('Time')
# plt.xlabel('Matrix size')
# plt.ylabel('Time, ms')
# plt.legend(['maxmod', 'delta_result', 'delta_x0'])
# plt.show()

n = 200
cur = 1e-16
count = 50
step = 1.5

x = np.zeros(count)
y = np.zeros(count)
for i in range(count):
    x[i] = cur

    A1 = np.random.random((n, n)) * 2 - 1 + np.identity(n) * (np.random.random((n, n)) + n + 1)
    b1 = np.random.random((n, 1)) * 2 - 1
    A2 = np.copy(A1)
    b2 = np.copy(b1)
    Atest = np.copy(A1)
    btest = np.copy(b1)

    x0 = np.linalg.solve(A1, b1)
    y[i] = solve_seidel(A2, b2, delta=cur)[1]

    cur *= step

plt.figure(figsize=(12, 8))
plt.semilogx(x, y)
plt.grid()
plt.title('Steps')
plt.xlabel('delta')
plt.ylabel('steps')
plt.legend(['maxmod'])
plt.show()
