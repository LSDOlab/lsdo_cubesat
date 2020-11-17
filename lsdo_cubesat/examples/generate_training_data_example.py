import matplotlib.pyplot as plt
import numpy as np
import pymesh
import trimesh
from matplotlib import cm
from numpy import savetxt
from lsdo_cubesat.solar.training.compute_solar_illumination import compute_solar_illumination
import os
here = os.path.dirname(os.path.abspath(__file__))

# This is here only because it takes a while to generate the data
quit()

# Number of angles to use for computing exposure (resolution)
n = 20

# Load PLY file with colored faces
mesh_stl = trimesh.load('cubesat_6u.ply')
mesh_stl_p = pymesh.load_mesh('cubesat_6u.ply')

# Get colors to differentiate between solar panel faces and other faces
face_colors = mesh_stl.visual.face_colors

# faces already constructed in generatemesh.py
mesh_stl_p.add_attribute("face_normal")
mesh_stl_p.add_attribute("face_area")
mesh_stl_p.add_attribute("face_centroid")
mesh_stl_p.add_attribute("face_index")

rmi = trimesh.ray.ray_triangle.RayMeshIntersector(mesh_stl)

# compute exposure for all azimuth and elevation angles
azimuth = np.linspace(-np.pi, np.pi, n)
elevation = np.linspace(-np.pi, np.pi, n)
illuminated_area = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        print(i, ', ', j)
        illuminated_area[i, j], solar_panels_area = compute_solar_illumination(
            azimuth[i],
            elevation[j],
            mesh_stl_p.get_face_attribute("face_area"),
            mesh_stl_p.get_face_attribute("face_normal"),
            mesh_stl_p.get_face_attribute("face_centroid"),
            face_colors,
            # rmi,
        )

# Save data as column vectors
x, y = np.meshgrid(azimuth, elevation)
xdata = np.array([x.flatten(), y.flatten()]).reshape(2, n**2)
savetxt(here + 'data/cubesat_xdata.csv', xdata[0, :], delimiter=',')
savetxt(here + 'data/cubesat_ydata.csv', xdata[1, :], delimiter=',')
savetxt(here + 'data/cubesat_zdata.csv',
        illuminated_area.flatten(),
        delimiter=',')

# Plot
fig = plt.figure()
ax = fig.add_subplot()
ax.contourf(x, y, illuminated_area)
ax.set_xlabel('azimuth [rad]')
ax.set_ylabel('elevation [rad]')
plt.show()
