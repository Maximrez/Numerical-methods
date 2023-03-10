import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from functions import *

if __name__ == '__main__':
    n_points, c, T, l = 400, 0.7, 10, 10
    k = 5

    method = special_LAX_WN
    phi = special_phi
    a_func = a_func1
    grid, matrix = method(phi, simple_mu, n_points, T, a_func1, c, l)
    frames = matrix.shape[0]
    # print(frames)
    # for i in range(len(matrix)):
    #     print(list(matrix[i]))

    fig = plt.figure(figsize=(16, 9))
    ax = plt.axes(xlim=(0, l), ylim=(-1.05, 1.05))
    line, = ax.plot([], [], lw=2)
    label = ax.text(l / 2, 1.12, 'summ=', ha='center', va='center', fontsize=20, color="Black")
    ax.grid()


    def init():
        line.set_data([], [])
        label.set_text('summ=')
        return line,


    def animate(i):
        line.set_data(grid, matrix[i])
        label.set_text('summ={0:.2f}'.format(sum(matrix[i])))
        return line,


    anim = FuncAnimation(fig, animate, init_func=init, frames=frames, interval=20, blit=True)

    anim.save(f'anim/{method.__name__}_{phi.__name__}_{a_func.__name__}.gif', writer='pillow')
    # plt.plot(grid, matrix[150])
    # plt.plot(grid, matrix[50])
    # print(list(matrix[150]))
    # print(list(matrix[50]))

    # plt.show()
