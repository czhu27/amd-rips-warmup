import numpy as np
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

f = 2.0
t0 = 0.3

t = np.linspace(0.0, 1.0, 1001)

f2 = np.square(2*np.pi*f)

r = (1 - f2*np.square(t - t0))*np.exp(-0.5*f2*np.square(t - t0))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(t, r)
ax.set(xlabel='time', ylabel='source amplitude')
plt.show()
