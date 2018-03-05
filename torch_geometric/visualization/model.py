import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa


def show_model(data, show_edges=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2])

    if show_edges:
        for i in range(data.index.size(1)):
            row, col = data.index[:, i]
            ax.plot(
                [data.pos[row][0], data.pos[col][0]],
                [data.pos[row][1], data.pos[col][1]],
                zs=[data.pos[row][2], data.pos[col][2]])

    plt.show()
