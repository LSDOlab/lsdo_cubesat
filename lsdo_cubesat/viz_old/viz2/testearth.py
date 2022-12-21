import mpl_toolkits
mpl_toolkits.__path__.append('/usr/lib/python2.7/dist-packages/mpl_toolkits/')
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
# set up orthographic map projection with
# perspective of satellite looking down at 50N, 100W.
# use low resolution coastlines.
# don't plot features that are smaller than 1000 square km.
bmap = Basemap(projection='ortho',
               lat_0=50,
               lon_0=-100,
               resolution='l',
               area_thresh=1000.)
# plot surface
bmap.warpimage(image='/home/lsdo/Cubesat/lsdo_cubesat/viz/blue_marble.jpg')
# draw the edge of the map projection region (the projection limb)
bmap.drawmapboundary()
# draw lat/lon grid lines every 30 degrees.
bmap.drawmeridians(np.arange(0, 360, 30))
bmap.drawparallels(np.arange(-90, 90, 30))
plt.show()