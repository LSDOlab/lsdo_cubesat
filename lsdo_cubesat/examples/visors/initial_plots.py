import matplotlib.pyplot as plt
import numpy as np


def plot_initial(sim):
    if sim.iter == 0:
        print('Need to run simulation before generating plots!')
    else:
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

        # zero when sun_LOS == 0
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

        plt.plot(sim['optics_cubesat_group.rw_torque'][0, :] -
                 sim['optics_cubesat_group.body_torque'][0, :])
        plt.plot(sim['optics_cubesat_group.rw_torque'][1, :] -
                 sim['optics_cubesat_group.body_torque'][1, :])
        plt.plot(sim['optics_cubesat_group.rw_torque'][2, :] -
                 sim['optics_cubesat_group.body_torque'][2, :])
        plt.plot(sim['detector_cubesat_group.rw_torque'][0, :] -
                 sim['detector_cubesat_group.body_torque'][0, :])
        plt.plot(sim['detector_cubesat_group.rw_torque'][1, :] -
                 sim['detector_cubesat_group.body_torque'][1, :])
        plt.plot(sim['detector_cubesat_group.rw_torque'][2, :] -
                 sim['detector_cubesat_group.body_torque'][2, :])
        plt.show()

        # # should NOT be constant
        plt.plot(sim['optics_cubesat_group.soc'].flatten())
        plt.show()

        # should be 2.4180232931306733 when soc << 0
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
