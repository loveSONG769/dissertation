import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(np.linspace(-5., 5., 100), np.linspace(-5., 5., 100))

def zfunc(x, y):
    return np.exp(-(y**2)) * np.cos(3.*x) + np.exp(x**2) * np.cos(3.*y)

z = zfunc(x, y)

ax.plot_surface(x, y, z)

plt.show()
