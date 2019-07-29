import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedFormatter
from mayavi.mlab import *
from mayavi import mlab
from matplotlib.patches import Rectangle
import main

sns.set()


def make_format_coord(ax, data):
    def format_coord(x, y):
        x1 = ax.format_xdata(x)
        y1 = ax.format_ydata(y)
        for x_i, y_i, z_i in data:
            if x_i == x1 and y_i == y1:
                return 'x=%s y=%s val=%s' % (x_i, y_i, z_i)
        return '???'
    return format_coord


class Formatter(FixedFormatter):
    def __init__(self, old):
        super().__init__(list(map(lambda x: x, old.seq)))

    def __call__(self, x, pos=None):
        return self.seq[abs(self.locs - x).argmin()]


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def draw(data):
    x, y, z = zip(*data)

    # r = np.array(data).reshape(main.N, main.N, 3)
    # res = []
    # for i in range(40):
    #     temp = []
    #     for j in range(40):
    #         temp.append(r[j][i][2])
    #     res.append(temp)
    # for i in range(len(res[0])):
    #     same = True
    #     for j in range(len(res) - 1):
    #         if not (np.array(res[j][i:]) == np.array(res[j + 1][i:])).all():  #  < [np.finfo(float).eps for _ in range(i)]:
    #             same = False
    #             break
    #     if same:
    #         print('123', res[0][i:])
    #         break

    z = np.array(z)

    # points = [a for a in list(zip(x, z)) if ~np.isnan(a[0]) and ~np.isnan(a[1])]
    # print(points)
    # print([abs((points[i][1] - points[i+1][1]) * (points[i][0] - points[i+2][0]) - (points[i][1] - points[i+2][1]) * (points[i][0] - points[i+1][0])) < np.finfo(float).eps for i in range(len(points) - 2)])

    # plt.plot(x[:-1], [z[i+1]-z[i] for i in range(99)])
    # plt.scatter(x, z)
    fz = lambda _x: (_x - x[0]) * (z[1] - z[0]) / (x[1] - x[0]) + z[0]
    fx = lambda _z: (_z - z[0]) * (x[1] - x[0]) / (z[1] - z[0]) + x[0]
    plt.plot([x[0], fz(0)], [z[0], 0])
    plt.plot(x, z)
    # df = pd.DataFrame.from_dict(np.array([x, y, z]).T)
    # df.columns = ['TAU', 'R', 'Q']
    # pivoted = df.pivot('R', 'TAU', 'Q')
    # plt.figure(figsize=(25.6, 13.43))
    # sns.heatmap(pivoted, cmap='coolwarm', annot=False, xticklabels=True, yticklabels=True, vmin=0, vmax=1, fmt='.2f')
    # plt.gca().format_coord = make_format_coord(plt.gca(), data)
    # plt.gca().xaxis.set_major_formatter(Formatter(plt.gca().xaxis.get_major_formatter()))
    # plt.gca().yaxis.set_major_formatter(Formatter(plt.gca().yaxis.get_major_formatter()))
    # colors = {-1: 'green', -2: 'pink'}
    # for i, row in enumerate(pivoted.values):
    #     for j, elem in enumerate(row):
    #         if elem < 0:
    #             plt.gca().add_patch(Rectangle((j, i),
    #                                           1, 1, fill=True, color=colors[elem]))
    #             pivoted.values[i][j] = np.nan
    # for i in range(pivoted.values.shape[1]):
    #     try:
    #         plt.gca().add_patch(Rectangle((i, np.nanargmin(pivoted.values[:, i])),
    #                                       1, 1, fill=False, edgecolor='blue', lw=3))
    #     except ValueError:
    #         pass
    # plt.gca().add_patch(Rectangle(np.unravel_index(np.nanargmin(pivoted.values.T), pivoted.values.T.shape),
    #                               1, 1, fill=False, edgecolor='yellow', lw=3))
    # plt.tight_layout()

    # from mpl_toolkits.mplot3d import axes3d, Axes3D
    # ax = Axes3D(plt.figure())
    # surf = ax.plot_trisurf(x,y[::-1],z)
    # plt.show()

    # pts = mlab.points3d(x, y[::-1], z, z, scale_mode='none', scale_factor=0)
    # mesh = mlab.pipeline.delaunay2d(pts)
    # surf = mlab.pipeline.surface(mesh)
    # mlab.show()


def draw_screen(data):
    draw(data)
    plt.show()
    exit()


def draw_file(data, filename):
    draw(data)
    plt.savefig(filename)


if __name__ == '__main__':
    with open("res", "rb") as data_f:
        draw_screen(pickle.load(data_f))
        # draw_file(pickle.load(data_f), 'out.png')
