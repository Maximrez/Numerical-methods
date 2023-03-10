from celluloid import Camera
import matplotlib.pyplot as plt

from functions import *

if __name__ == "__main__":
    n_points, c, T, l, a = 400, 0.5, 2.5, 10, 1
    k = 5
    # print(int(T * a * (n_points - 1) / (c * l)))

    phis = [phi1, phi2, phi3, phi4]
    method = LAXTVD  # square angle LAX_WN LAXTVD
    for phi in phis:
        grid1, matrix1 = method(phi, simple_mu, n_points, T, c, l, a)
        grid2, matrix2 = method(phi, simple_mu, k * n_points, T, c, l, a)
        h1 = l / (n_points - 1)
        h2 = l / (k * n_points - 1)
        exact_matrix1 = exact(phi, simple_mu, n_points, T, c, l, a)
        exact_matrix2 = exact(phi, simple_mu, k * n_points, T, c, l, a)

        n_frames1 = matrix1.shape[0]
        n_frames2 = matrix2.shape[0]
        alpha = n_frames2 / n_frames1

        eps1 = np.zeros(n_frames1)
        eps2 = np.zeros(n_frames1)
        for i in range(n_frames1):
            eps1[i] = np.max(np.abs(exact_matrix1[i] - matrix1[i]))
            eps2[i] = np.max(np.abs(exact_matrix2[int(alpha * i)] - matrix2[int(alpha * i)]))

        eps1[0] += 1e-10
        eps2[0] += 1e-10
        tau = c * l / (n_points - 1) / a
        time_axis = np.arange(0, n_frames1) * tau
        fig, axs = plt.subplots(2, figsize=(16, 9))
        axs[0].set(xlim=(0, l), ylim=(0, 1.05), ylabel='y')
        max_eps = np.max(eps1)
        axs[1].set(xlim=(0, n_frames1 * tau), ylim=(0, max_eps * 1.05), xlabel='Time', ylabel='eps')
        axs[0].grid()
        axs[1].grid()
        camera = Camera(fig)

        for i in range(n_frames1):
            axs[0].plot(grid1, matrix1[i], color='blue')
            axs[0].plot(grid2, matrix2[int(alpha * i)], color='green')
            axs[0].plot(grid1, exact_matrix1[i], color='orange')
            axs[0].legend([str(round(h1, 3)), str(round(h2, 3)), 'Exact matrix'])

            axs[1].plot(time_axis[:i], eps1[:i], color='blue')
            axs[1].plot(time_axis[:i], eps2[:i], color='green')

            axs[0].annotate('С = {0:.1f}, Время t = {1:.2f}'.format(c, i * tau), (l / 2, 1.15), size=16, ha='center', annotation_clip=False)
            axs[1].annotate(r'$ Отношение \: погрешностей: \epsilon_{1} / \epsilon_{2} = {%.1f} $' % (eps1[i] / eps2[i]), (T / 2, 1.12 * max_eps),
                            size=16, ha='center', annotation_clip=False)
            axs[1].legend([str(round(h1, 3)), str(round(h2, 3))])
            camera.snap()

        anim = camera.animate()
        print(f'Saving: {method.__name__}_{phi.__name__}.gif')
        anim.save(f'anim/{method.__name__}_{phi.__name__}.gif', writer='pillow', fps=30)
