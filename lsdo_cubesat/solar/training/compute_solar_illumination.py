import numpy as np


def compute_solar_illumination(
    azimuth,
    elevation,
    faces_area,
    face_normal,
    face_centroid,
    face_colors,
    rmi=None,
):
    """
    Compute solar illumination. Use this function to generate training
    date from a CAD model in PLY format with solar panel triangles
    colored blue and all other triangles colored green.

    Parameters
    ----------
    azimuth : float
        Angle of azimuth in radians
    elevation : float
        Angle of elevation in radians
    face_area
        The ``face_area`` attribute of a PyMesh object
    normals
        The ``face_normal`` attribute of a PyMesh object
    centroids
        The ``face_centroid`` attribute of a PyMesh object
    face_colors : trimesh.Trimesh.visual.face_colors
        Colors used to distinguish between triangles corresponding to
        solar panels and other triangles
    rmi : RayMeshIntersector
        An instance of the trimesh RayMeshIntersector class, used to
        detect shadows cast by spacecraft onto solar panels. If the
        geometry is such that spacecraft does not cast a shadow onto any
        solar panel, omitting this argument will result in faster
        generation of training data without any loss in accuracy.

    Returns
    -------
    illuminated_area : array
        total area illuminated as a function of azimuth and elevation
    solar_panels_area : float
        total area used to normalize illumination
    """

    # compute vector from spacecraft to sun
    Ax = np.cos(azimuth) * np.cos(elevation)
    Ay = np.sin(azimuth) * np.cos(elevation)
    Az = np.sin(elevation)
    print(Ax, Ay, Az)
    print(azimuth * 180 / np.pi, elevation * 180 / np.pi)
    sc_to_sun = np.array([Ax, Ay, Az])[np.newaxis]
    sc_to_sun /= np.linalg.norm(sc_to_sun, axis=1)

    TOTAL_NUM_POLYGONS = len(faces_area)

    # iterate over face normals
    illuminated_area = 0
    solar_panels_area = 0
    for i in range(TOTAL_NUM_POLYGONS):
        solar_panel = False
        # face is more blue than green (no check for red value)
        if face_colors[i][2] > face_colors[i][1]:
            solar_panel = True

        # compute and sum illumination on all solar panels
        if solar_panel == True:
            solar_panels_area += faces_area[i]
            face_normal = face_normal[i, :]  # panel normals
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
                            face_centroid[i][np.newaxis] + \
                            np.sign(face_normal[i]) * 1e-5,
                            sc_to_sun,
                        )

                # update illumination
                if shadow != [True]:
                    illuminated_area += mag_normal_to_sun * faces_area[i]

    return (illuminated_area, solar_panels_area)
