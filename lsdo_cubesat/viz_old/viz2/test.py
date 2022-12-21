import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# detector_data = np.fromfile(
#     '/home/lsdo/Cubesat/lsdo_cubesat/viz/detector_position.dat', dtype=float)
# print(detector_data.shape)

data = np.loadtxt('/home/lsdo/Cubesat/lsdo_cubesat/viz/detector_position.dat',
                  skiprows=3)
print(data[:, 0])

plt.plot(data[:, 0], data[:, 1])
plt.show()

# names = ["X", "Y", "Z"]
# data = np.genfromtxt(
#     '/home/lsdo/Cubesat/lsdo_cubesat/viz/detector_position.dat',
#     dtype=None,
#     names=names,)
# print(data)

# data = pd.read_csv('/home/lsdo/Cubesat/lsdo_cubesat/viz/detector_position.dat',
#                    delim_whitespace=True)
# print(data[0])
