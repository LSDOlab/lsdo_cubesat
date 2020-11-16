Solar Illumination
==================

Raytracing
----------

compute_solar_illumination is a function used when generating training
data for the surrogate model.


.. code-block:: python

  import numpy as np
  
  
  def compute_solar_illumination(
      azimuth,
      elevation,
      faces_areas,
      normals,
      centroids,
      face_colors,
      rmi=None,
  ):
      """
      Compute solar illumination
      """
  
      # compute vector from spacecraft to sun
      Ax = np.cos(azimuth) * np.cos(elevation)
      Ay = np.sin(azimuth) * np.cos(elevation)
      Az = np.sin(elevation)
      print(Ax, Ay, Az)
      print(azimuth * 180 / np.pi, elevation * 180 / np.pi)
      sc_to_sun = np.array([Ax, Ay, Az])[np.newaxis]
      sc_to_sun /= np.linalg.norm(sc_to_sun, axis=1)
  
      TOTAL_NUM_POLYGONS = len(faces_areas)
  
      # iterate over face normals
      sunlit_area = 0
      solar_panels_area = 0
      for i in range(TOTAL_NUM_POLYGONS):
          solar_panel = False
          # face is more blue than green (no check for red value)
          if face_colors[i][2] > face_colors[i][1]:
              solar_panel = True
  
          # compute and sum illumination on all solar panels
          if solar_panel == True:
              solar_panels_area += faces_areas[i]
              face_normal = normals[i, :]  # panel normals
              mag_normal_to_sun = np.dot(sc_to_sun, face_normal)
              panel_faces_sun = mag_normal_to_sun > 0
              if (panel_faces_sun == True):
                  shadow = False
                  # Checking if face is being struck by shadow
                  if rmi is not None:
                      for m in range(TOTAL_NUM_POLYGONS):
                          # need to offset centroid from face so that ray
                          # doesn't intersect the current face
                          shadow = shadow or rmi.intersects_any(
                              centroids[i][np.newaxis] + \
                              np.sign(normals[i]) * 1e-5,
                              sc_to_sun,
                          )
  
                  # update illumination
                  if shadow != [True]:
                      sunlit_area += mag_normal_to_sun * faces_areas[i]
  
      return (sunlit_area, solar_panels_area)
  
