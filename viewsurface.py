import numpy as np
import matplotlib.pyplot as plt


def vsurface(f, x1, x2):
    xgrid, ygrid = np.meshgrid(x1, x2)
    # xy = np.stack([xgrid, ygrid])
    xy = np.zeros((xgrid.size, 2))
    xgrid1 = xgrid.reshape((xgrid.size,))
    ygrid1 = ygrid.reshape((ygrid.size,))
    count = 0
    for i in range(xgrid1.size):
        xy[count, 0] = xgrid1[i]
        xy[count, 1] = ygrid1[i]
        count += 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, -45)
    ax.plot_surface(xgrid, ygrid, f(xy).reshape(xgrid.shape), cmap='terrain')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    plt.show()


if __name__ == '__main__':
    from benchmarks import schaffer2, manyf_settings as f_settings

    symbol = "schaffer2"
    settings = f_settings(dims=2)
    f = eval(symbol)

    bounds = settings[symbol]["bounds"]
    step = (bounds[0][1] - bounds[0][0]) / 200
    # f= ""

    x = np.arange(bounds[0][0], bounds[0][1], step=step)
    y = np.arange(bounds[1][0], bounds[1][1], step=step)
    f = f

    vsurface(f, x1=x, x2=y)
