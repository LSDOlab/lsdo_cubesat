import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pyproj
import pickle
import pandas as pd
import seaborn as sns

from PIL import Image


def XYZ_2_LLA(X, Y, Z):

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    lon, lat, alt = pyproj.transform(ecef, lla, X, Y, Z, radians=True)

    lon = 180 / np.pi * lon
    lat = 180 / np.pi * lat
    alt = 180 / np.pi * alt

    return lon, lat, alt


def datasort(lon, lat, alt):
    A = np.array([lon, lat, alt])
    B = A.T
    C = B[np.lexsort(B[:, ::-1].T)]
    return C


def viz(X, Y, Z):
    lon, lat, alt = XYZ_2_LLA(X, Y, Z)
    matrix = datasort(lon, lat, alt)
    return matrix


x = 652954.1006
y = 4774619.7919
z = -2217647.7937

gs_lon = -83.7264
gs_lat = 42.2708
gs_alt = 0.256

ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

path = '/home/lsdo/Cubesat/lsdo_cubesat/_data/opt.00219.pkl'

with open(path, 'rb') as f:
    info = pickle.load(f)
# print(info['sunshade_cubesat_group.relative_orbit_state'].shape)

X_reference = info['reference_orbit_state'][0, :]
Y_reference = info['reference_orbit_state'][1, :]
Z_reference = info['reference_orbit_state'][2, :]

X_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][0, :]
Y_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][1, :]
Z_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][2, :]

X_detector_relative = info['detector_cubesat_group.relative_orbit_state'][0, :]
Y_detector_relative = info['detector_cubesat_group.relative_orbit_state'][1, :]
Z_detector_relative = info['detector_cubesat_group.relative_orbit_state'][2, :]

X_optics_relative = info['optics_cubesat_group.relative_orbit_state'][0, :]
Y_optics_relative = info['optics_cubesat_group.relative_orbit_state'][1, :]
Z_optics_relative = info['optics_cubesat_group.relative_orbit_state'][2, :]

# print(X_reference.shape)

X_detector = X_reference + X_detector_relative * 1e5
Y_detector = Y_reference + Y_detector_relative * 1e5
Z_detector = Z_reference + Z_detector_relative * 1e5

matrix_reference = viz(X_reference, Y_reference, Z_reference)
print(matrix_reference[:, 2])
matrix_detector = viz(X_detector, Y_detector, Z_detector)

grid = plt.GridSpec(2, 4, wspace=0.5, hspace=0.5)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:3])
plt.subplot(grid[0, 3])
plt.subplot(grid[1, 0])
plt.subplot(grid[1, 1:3])
plt.subplot(grid[1, 3])
plt.show()

# plt.figure()
# sns.set()
# plt.plot(matrix_reference[:, 0], matrix_reference[:, 2])
# plt.show()

# lon, lat, alt = pyproj.transform(ecef, lla, X, Y, Z, radians=True)
# lon = 180 / np.pi * lon
# lat = 180 / np.pi * lat

# A = np.array([lon, lat])

# B = A.T
# C = B[np.lexsort(B[:, ::-1].T)]
# print(C)

path = "/home/lsdo/Cubesat/lsdo_cubesat/map/world.jpg"
earth = mpimg.imread(path)

img = Image.open(path)
# print(img.size)
fig, ax = plt.subplots()
sns.set()
ax.imshow(earth, extent=[-180, 180, -100, 100])
ax.plot(matrix_reference[:, 0],
        matrix_reference[:, 1],
        linewidth='1',
        color='yellow')
# ax.scatter(gs_lon, gs_lat, marker="p", label="Michigen")
ax.scatter(-117.2340, 32.8801, marker="p", label="UCSD")
ax.scatter(-88.2272, 40.1020, marker="p", label="UIUC")
ax.scatter(-84.3963, 33.7756, marker="p", label="Georgia")
ax.scatter(-109.533691, 46.9653, marker="p", label="Montana")

# ax.plot(matrix_detector[:, 0],
#         matrix_detector[:, 1],
#         linewidth='1',
#         color='green')
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Trajectory of VISORS Satellite")
# ax.plot(np.arange(lon.shape[0]),lon)
# ax.scatter(C[:,0],C[:,1], marker='.', color='yellow')
plt.show()

# sns.set()
# optimal = np.array([
#     1.0e-3, 1.0e-3, 1.0e-3, 5.5e-5, 7.6e-5, 7.6e-5, 7.6e-5, 7.7e-5, 7.7e-5,
#     1.7e-4, 1.1e-4, 1.1e-4, 1.7e-5, 7.9e-6
# ])

# iteration = np.arange(np.shape(optimal)[0])
# plt.hlines(1e-5, 0, 13, colors='blue', linestyles="dashed", label="1e-5")

# plt.xlabel("Major iteration")
# plt.ylabel("Optimality")
# # fig, ax = plt.subplot()
# plt.plot(iteration, optimal)
# plt.show()