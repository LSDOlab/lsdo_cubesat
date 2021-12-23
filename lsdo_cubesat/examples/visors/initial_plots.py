import matplotlib.pyplot as plt
import numpy as np


def plot_initial(sim, threeD=False):
    if threeD:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        x = sim['reference_orbit_state_km'][0, :]
        y = sim['reference_orbit_state_km'][1, :]
        z = sim['reference_orbit_state_km'][2, :]
        l = min(np.min(x), np.min(y), np.min(z))
        u = max(np.max(x), np.max(y), np.max(z))
        ax.set_xlim(l, u)
        ax.set_ylim(l, u)
        ax.set_zlim(l, u)
        ax.plot(x, y, z)

        sd = sim['sun_direction']
        ax.plot(
            sd[0, :] * 6000,
            sd[1, :] * 6000,
            sd[2, :] * 6000,
        )
        plt.show()

    else:
        sim.run()

        plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][0, :])
        plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][1, :])
        plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][2, :])
        plt.show()

        # shows observation phase for each cubesat
        plt.plot(sim['optics_observation_dot'])
        plt.plot(sim['detector_observation_dot'])
        plt.plot(sim['optics_observation_phase_indicator'])
        plt.plot(sim['detector_observation_phase_indicator'])
        plt.show()

        # shows observation phase
        plt.plot(sim['observation_phase_indicator'])
        plt.show()

        # observation phase is within phase where sun_LOS==1
        plt.plot(sim['optics_cubesat_group.sun_LOS'][0, :])
        plt.plot(sim['detector_cubesat_group.sun_LOS'][0, :])
        plt.show()

        plt.plot(sim['optics_cubesat_group.percent_exposed_area'][0, :])
        plt.plot(sim['detector_cubesat_group.percent_exposed_area'][0, :])
        plt.show()

        # these are zero when Euler angles are zero
        plt.plot(sim['optics_cubesat_group.body_rates'][0, :])
        plt.plot(sim['optics_cubesat_group.body_rates'][1, :])
        plt.plot(sim['optics_cubesat_group.body_rates'][2, :])
        plt.show()

        # torques counteract each other when Euler angles are zero
        plt.plot(sim['optics_cubesat_group.rw_torque'][0, :])
        plt.plot(sim['optics_cubesat_group.rw_torque'][1, :])
        plt.plot(sim['optics_cubesat_group.rw_torque'][2, :])
        plt.show()

        plt.plot(sim['optics_cubesat_group.body_torque'][0, :])
        plt.plot(sim['optics_cubesat_group.body_torque'][1, :])
        plt.plot(sim['optics_cubesat_group.body_torque'][2, :])
        plt.show()

        # # should NOT be constant
        plt.plot(sim['optics_cubesat_group.soc'].flatten())
        plt.show()

        # should be 3 when soc <= 0
        plt.plot(sim['optics_cubesat_group.voltage'].flatten())
        plt.plot(3 + (np.exp(sim['optics_cubesat_group.soc'].flatten()) - 1) /
                 (np.exp(1) - 1))
        plt.show()

        # differ by a constant factor, ns*np
        plt.plot(sim['optics_cubesat_group.power'].flatten())
        plt.show()

        plt.plot(sim['optics_cubesat_group.battery_power'])
        plt.show()

        # plt.plot(sim['telescope_view_angle_error'].flatten())
        # plt.show()

        # # between 0 and 1
        # plt.plot(sim['optics_cos_view_angle_error'].flatten())
        # plt.plot(sim['optics_cos_view_angle_error_during_observation'].flatten())
        # plt.show()

        # # arcsec
        # plt.plot(sim['optics_view_angle_error'].flatten())
        # plt.show()
        # plt.plot(sim['detector_view_angle_error'].flatten())
        # plt.show()
        exit()