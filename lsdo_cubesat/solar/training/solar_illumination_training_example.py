import matplotlib.pyplot as plt
from lsdo_cubesat.solar.smt_exposure import smt_exposure
import numpy as np
from functools import reduce

from smt.surrogate_models import RMTB

# load training data
from lsdo_cubesat.solar.training.data.cubesat_xdata import cubesat_xdata as az
from lsdo_cubesat.solar.training.data.cubesat_ydata import cubesat_ydata as el
from lsdo_cubesat.solar.training.data.cubesat_zdata import cubesat_zdata as yt

n = 100
azimuth = np.linspace(-np.pi, np.pi, 2 * n)
elevation = np.linspace(-np.pi / 2, np.pi / 2, n)


def structure_data(*args):
    inputs = np.meshgrid(*args, indexing='ij')
    nx = len(inputs)
    nt = np.prod(inputs[0].shape)
    xt = np.array([x.flatten() for x in inputs]).T
    return xt, inputs


xt, inputs = structure_data(azimuth, elevation)
# yt = reduce(lambda x, y: x.flatten() * y.flatten(), inputs,
#             np.ones(inputs[0].shape)).T
X = inputs[0]
Y = inputs[1]
print(X.shape, Y.shape)

Z = X**2 * Y
yt = Z.flatten()

xlimits = np.array([
    [np.min(azimuth), np.max(azimuth)],
    [np.min(elevation), np.max(elevation)],
])
sm = RMTB(
    xlimits=xlimits,
    order=4,
    num_ctrl_pts=20,
    energy_weight=1e-3,
    regularization_weight=1e-7,
    print_global=False,
)

sm.set_training_values(xt, yt)
sm.train()

fig, ax = plt.subplots(1, 4)
CS = ax[0].contourf(X, Y, Z)
# plt.show()

# TODO: figure out max value of yt
n = 10

# plot 1 will show training data
# x - azimuth
# y - elevation
# contours - illumination
# CS = ax[0].contourf(
#     az.reshape((20, 20)),
#     el.reshape((20, 20)),
#     yt.reshape((20, 20)),
# )
# CS.cmap.set_under('black')
# CS.cmap.set_over('white')
# ax[0].clabel(CS, inline=1, fontsize=10)
ax[0].clabel(CS)
ax[0].set_title('training data')

# generate surrogate model
step = 1

# structure data FOR PREDICTION
az = np.linspace(-np.pi, np.pi, n)
el = np.linspace(-np.pi / 2, np.pi / 2, n)
x, y = np.meshgrid(az, el, indexing='ij')
rp = np.concatenate(
    (
        x.reshape(n**2, 1),
        y.reshape(n**2, 1),
    ),
    axis=1,
)

# generate predictions
sunlit_area = sm.predict_values(rp)
# if np.min(sunlit_area) < 0:
#     sunlit_area -= np.min(sunlit_area)
# else:
#     sunlit_area += np.min(sunlit_area)
# max_sunlit_area = min(1, np.max(sunlit_area))
# sunlit_area /= np.max(sunlit_area)
# sunlit_area *= max_sunlit_area

dzdx = sm.predict_derivatives(
    rp,
    0,
)

dzdy = sm.predict_derivatives(
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
    dzdx.reshape((n, n)),
)
ax[2].set_title('prediction (dz/dx)')
CS = ax[3].contourf(
    x.reshape((n, n)),
    y.reshape((n, n)),
    dzdy.reshape((n, n)),
)
ax[3].set_title('prediction (dz/dy)')
fig = plt.gcf()
fig.set_size_inches(12, 6)
plt.show()
