import matplotlib.pyplot as plt
from lsdo_cubesat.solar.smt_exposure import smt_exposure
import numpy as np
import os
here = os.path.dirname(os.path.abspath(__file__))

n = 10

# load training data
az = np.genfromtxt(here + '/data/cubesat_xdata.csv', delimiter=',')
el = np.genfromtxt(here + '/data/cubesat_ydata.csv', delimiter=',')
yt = np.genfromtxt(here + '/data/cubesat_zdata.csv', delimiter=',')

fig, ax = plt.subplots(1, 4)
CS = ax[0].contourf(
    az.reshape((20, 20)),
    el.reshape((20, 20)),
    yt.reshape((20, 20)),
)
CS.cmap.set_under('black')
CS.cmap.set_over('white')
# ax[0].clabel(CS, inline=1, fontsize=10)
ax[0].clabel(CS)
ax[0].set_title('training data')

# generate surrogate model
step = 1
sm = smt_exposure(
    int(len(yt) / step),
    az[::step],
    el[::step],
    yt[::step],
)

# generate predictions
n = 800
az = np.linspace(-np.pi, np.pi, n)
el = np.linspace(-np.pi, np.pi, n)
x, y = np.meshgrid(az, el, indexing='xy')
rp = np.concatenate(
    (
        x.reshape(n**2, 1),
        y.reshape(n**2, 1),
    ),
    axis=1,
)
sunlit_area = np.zeros(n**2).reshape((n, n))
sunlit_area = sm.predict_values(rp)
if np.min(sunlit_area) < 0:
    sunlit_area -= np.min(sunlit_area)
else:
    sunlit_area += np.min(sunlit_area)
max_sunlit_area = min(1, np.max(sunlit_area))
sunlit_area /= np.max(sunlit_area)
sunlit_area *= max_sunlit_area

dadx = sm.predict_derivatives(
    rp,
    0,
)

dady = sm.predict_derivatives(
    rp,
    1,
)

CS = ax[1].contourf(
    x.reshape((n, n)),
    y.reshape((n, n)),
    sunlit_area.reshape((n, n)),
)
ax[1].set_title('prediction')
CS = ax[2].contourf(
    x.reshape((n, n)),
    y.reshape((n, n)),
    dadx.reshape((n, n)),
)
ax[2].set_title('prediction (dz/dx)')
CS = ax[3].contourf(
    x.reshape((n, n)),
    y.reshape((n, n)),
    dady.reshape((n, n)),
)
ax[3].set_title('prediction (dz/dy)')
fig = plt.gcf()
fig.set_size_inches(12, 6)
plt.show()
