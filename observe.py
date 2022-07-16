import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from evolutionary.de.benchmarks import schaffer4, manyf_settings as f_settings
import numpy as np

symbol = "schaffer2"
settings = f_settings(dims=2)
bounds = settings[symbol]["bounds"]
step = (bounds[0][1]-bounds[0][0])/200
x = np.arange(bounds[0][0], bounds[0][1], step=step)
y = np.arange(bounds[1][0], bounds[1][1], step=step)
f = eval(symbol)

xgrid, ygrid = np.meshgrid(x, y)
# xy = np.stack([xgrid, ygrid])
xy = np.zeros((xgrid.size, 2))
xgrid1 = xgrid.reshape((xgrid.size,))
ygrid1 = ygrid.reshape((ygrid.size,))
count = 0
for i in range(xgrid1.size):
    xy[count, 0] = xgrid1[i]
    xy[count, 1] = ygrid1[i]
    count +=1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(45, -45)
ax.plot_surface(xgrid, ygrid, f(xy).reshape(xgrid.shape), cmap='terrain')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('{}(x, y)'.format(symbol))
plt.show()