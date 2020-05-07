from itertools import compress

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from Prototype.kruskal_clustering import FullGraph



def main():

    p1 = np.random.normal(0, 100, (30, 3))
    p2 = np.random.normal(300, 100, (30, 3))
    p3 = np.random.normal(600, 100, (30, 3))

    points = np.append(p1, p2, axis=0)
    points = np.append(points, p3, axis=0)

    g = FullGraph(points)
    edges = list(compress(g.E, g.EChoose))

    def draw_line(n, points, lines):
        line = lines[n]
        i, j, _ = edges[n]
        coords = points[[i, j]].transpose()
        line.set_data_3d(coords)
        return lines

    fig = plt.figure(figsize=(8.5, 18), dpi=100, constrained_layout=True)
    plt.suptitle("MST Clustering")
    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[5, 1])

    ax = fig.add_subplot(spec[0, 0], projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.scatter(*zip(*p1), color="tab:red")
    ax.scatter(*zip(*p2), color="tab:blue")
    ax.scatter(*zip(*p3), color="tab:green")

    lines = [ax.plot([], [], [])[0] for _ in range(len(edges))]

    anim = animation.FuncAnimation(
        fig, draw_line, len(edges), fargs=(points, lines), interval=50, blit=True
    )

    ax = fig.add_subplot(spec[1, 0])
    ax.set_xlabel("Edge")
    ax.set_ylabel("Max Partition Size")
    ax.plot(range(1, len(edges) + 1), g.MaxPartitionEvolve)
    ax.set_xlim(1, len(edges))

    bump_idx = np.argmax(np.diff(g.MaxPartitionEvolve)) + 2
    ax.axvline(bump_idx, color="tab:orange")
    ax.set_xticks(list(ax.get_xticks()) + [bump_idx])


    # Show the plot, save the animation, or save the figure
    # anim.save("kruskal-clustering.mp4")
    plt.show()

    #plt.savefig("kruskal-clustering.png")


if __name__ == "__main__":
    main()