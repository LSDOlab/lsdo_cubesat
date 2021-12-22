from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.parameters.cubesat import CubesatParams
from lsdo_cubesat.parameters.ground_station import GroundStationParams
import numpy as np

num_times = 30
num_cp = 3
# step_size = 50.
step_size = 95 * 60 / (num_times - 1)

swarm = SwarmParams(
    num_times=num_times,
    num_cp=num_cp,
    step_size=step_size,
    cross_threshold=0.882,
)

# initial_orbit_state_magnitude = np.array([0.001] * 3 + [0.001] * 3)

initial_orbit_state_magnitude = np.array([1e-3] * 3 + [1e-3] * 3)

np.random.seed(6)
# A = np.random.rand(6)

print(initial_orbit_state_magnitude * np.random.rand(6))

cubesats = dict()
cubesats['sunshade'] = CubesatParams(
    name='sunshade',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    apogee_altitude=500.001,
    perigee_altitude=499.99,
)

cubesats['optics'] = CubesatParams(
    name='optics',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    perigee_altitude=500.,
    apogee_altitude=500.,
)

cubesats['detector'] = CubesatParams(
    name='detector',
    dry_mass=1.3,
    initial_orbit_state=initial_orbit_state_magnitude * np.random.rand(6),
    approx_altitude_km=500.,
    specific_impulse=47.,
    perigee_altitude=500.002,
    apogee_altitude=499.98,
)

cubesats['sunshade'].add(
    GroundStationParams(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
cubesats['sunshade'].add(
    GroundStationParams(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
cubesats['sunshade'].add(
    GroundStationParams(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
cubesats['sunshade'].add(
    GroundStationParams(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

cubesats['detector'].add(
    GroundStationParams(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
cubesats['detector'].add(
    GroundStationParams(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
cubesats['detector'].add(
    GroundStationParams(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
cubesats['detector'].add(
    GroundStationParams(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

cubesats['optics'].add(
    GroundStationParams(
        name='UCSD',
        lon=-117.1611,
        lat=32.7157,
        alt=0.4849,
    ))
cubesats['optics'].add(
    GroundStationParams(
        name='UIUC',
        lon=-88.2272,
        lat=32.8801,
        alt=0.2329,
    ))
cubesats['optics'].add(
    GroundStationParams(
        name='Georgia',
        lon=-84.3963,
        lat=33.7756,
        alt=0.2969,
    ))
cubesats['optics'].add(
    GroundStationParams(
        name='Montana',
        lon=-109.5337,
        lat=33.7756,
        alt=1.04,
    ))

for v in cubesats.values():
    swarm.add(v)
