from csdl_om import Simulator
from lsdo_cubesat.orbit.reference_orbit import ReferenceOrbit
import numpy as np
import matplotlib.pyplot as plt


def generate_reference_orbit(num_times, step_size, plot=False):
    sim = Simulator(ReferenceOrbit(
        num_times=num_times,
        step_size=step_size,
    ))
    sim.run()
    if plot is True:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # plot trajectory
        x = sim['reference_orbit_state_km'][0, :]
        y = sim['reference_orbit_state_km'][1, :]
        z = sim['reference_orbit_state_km'][2, :]
        l = min(np.min(x), np.min(y), np.min(z))
        u = max(np.max(x), np.max(y), np.max(z))
        ax.set_xlim(l, u)
        ax.set_ylim(l, u)
        ax.set_zlim(l, u)
        ax.plot(x, y, z)

        # plot initial state
        r0 = sim['initial_radius_km']
        rx = r0[0]
        ry = r0[1]
        rz = r0[2]
        v0 = sim['initial_velocity_km_s']
        v0 /= np.linalg.norm(v0)
        vx = v0[0]
        vy = v0[1]
        vz = v0[2]
        k = 3000.
        r_vec = np.array([
            [0, rx],
            [0, ry],
            [0, rz],
        ])
        ax.plot(
            r_vec[0, :],
            r_vec[1, :],
            r_vec[2, :],
        )
        vel_vec = np.array([
            [rx, rx + k * vx],
            [ry, ry + k * vy],
            [rz, rz + k * vz],
        ])
        ax.plot(
            vel_vec[0, :],
            vel_vec[1, :],
            vel_vec[2, :],
        )
        return sim['reference_orbit_state_km'], ax
    return sim['reference_orbit_state_km'], None
