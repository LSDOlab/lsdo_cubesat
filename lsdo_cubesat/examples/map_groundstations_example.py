import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pyproj
import pickle
import pandas as pd
import seaborn as sns

from PIL import Image

from lsdo_cubesat.options.ground_station import Ground_station

ucsd = Ground_station(
    name='UCSD',
    lon=-117.1611,
    lat=32.8801,
    alt=0.4849,
)
uiuc = Ground_station(
    name='UIUC',
    lon=-88.2272,
    lat=32.8801,
    alt=0.2329,
)
gt = Ground_station(
    name='Georgia',
    lon=-84.3963,
    lat=33.7756,
    alt=0.2969,
)
mtu = Ground_station(
    name='Montana',
    lon=-109.5337,
    lat=33.7756,
    alt=1.04,
)
mich = Ground_station(
    name='Michigan',
    lon=-83.7264,
    lat=42.2708,
    alt=0.2329,
)

ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

grid = plt.GridSpec(2, 4, wspace=0.5, hspace=0.5)
plt.subplot(grid[0, 0])
plt.subplot(grid[0, 1:3])
plt.subplot(grid[0, 3])
plt.subplot(grid[1, 0])
plt.subplot(grid[1, 1:3])
plt.subplot(grid[1, 3])
plt.show()

path = "/Users/victor/packages/lsdo_cubesat/lsdo_cubesat/map/world.jpg"
earth = mpimg.imread(path)

img = Image.open(path)
fig, ax = plt.subplots()
sns.set()
ax.imshow(earth, extent=[-180, 180, -100, 100])
ax.scatter(mich['lon'], mich['lat'], marker="p", label=mich['name'])
ax.scatter(ucsd['lon'], ucsd['lat'], marker="p", label=ucsd['name'])
ax.scatter(uiuc['lon'], uiuc['lat'], marker="p", label=uiuc['name'])
ax.scatter(gt['lon'], gt['lat'], marker="p", label=gt['name'])
ax.scatter(mtu['lon'], mtu['lat'], marker="p", label=mtu['name'])
plt.xlabel("longitude")
plt.ylabel("latitude")
plt.title("Groundstation Locations")
plt.show()
