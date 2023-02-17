from lsdo_cubesat.specifications.cubesat_spec import CubesatSpec
from lsdo_cubesat.examples.visors_baseline.telescope import Telescope

import numpy as np
from csdl import GraphRepresentation


def make_swarm(swarm):

    np.random.seed(0)

    cubesats = dict()

    cubesats['optics'] = CubesatSpec(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=np.array([
            40,
            0,
            0,
            0.,
            0.,
            0.,
            # 1.76002146e+03,
            # 6.19179823e+03,
            # 6.31576531e+03,
            # 4.73422022e-05,
            # 1.26425269e-04,
            # 5.39731211e-05,
        ]),
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    )

    cubesats['detector'] = CubesatSpec(
        name='detector',
        dry_mass=1.3,
        initial_orbit_state=np.array([
            0,
            0,
            0,
            0,
            0.,
            0.,
            # 1.76002146e+03,
            # 6.19179823e+03,
            # 6.31576531e+03,
            # 4.73422022e-05,
            # 1.26425269e-04,
            # 5.39731211e-05,
        ]),
        specific_impulse=47.,
        perigee_altitude=500.002,
        apogee_altitude=499.98,
    )

        swarm.add(v)
    m = Telescope(swarm=swarm)
    return GraphRepresentation(m)
