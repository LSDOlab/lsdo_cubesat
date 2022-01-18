import matplotlib.pyplot as plt
import numpy as np


def plot_sim_state(sim):
    if sim.iter == 0:
        print('Need to run simulation before generating plots!')
    else:
        plt.title('sun_direction, x, y')
        plt.plot(sim['sun_direction'][0, :])
        plt.plot(sim['sun_direction'][1, :])
        plt.show()

        plt.title('attitude (optics)')
        plt.plot(sim['optics_cubesat_group.yaw'])
        plt.plot(sim['optics_cubesat_group.pitch'])
        plt.plot(sim['optics_cubesat_group.roll'])
        plt.show()

        plt.title('attitude (detector)')
        plt.plot(sim['detector_cubesat_group.yaw'])
        plt.plot(sim['detector_cubesat_group.pitch'])
        plt.plot(sim['detector_cubesat_group.roll'])
        plt.show()

        plt.title('relative_orbit_state_m')
        plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][0, :])
        plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][1, :])
        plt.plot(sim['optics_cubesat_group.relative_orbit_state_m'][2, :])
        plt.show()

        plt.title(
            'observation phase for each cubesat\nmust overlap for telescope constraints to be active'
        )
        plt.plot(sim['optics_observation_dot'])
        plt.plot(sim['detector_observation_dot'])
        plt.plot(sim['optics_observation_phase_indicator'])
        plt.plot(sim['detector_observation_phase_indicator'])
        plt.show()

        plt.title('thrust profile (ctrl pts)')
        plt.plot(sim['optics_cubesat_group.thrust_cp'][0, :])
        plt.plot(sim['optics_cubesat_group.thrust_cp'][1, :])
        plt.plot(sim['optics_cubesat_group.thrust_cp'][2, :])
        plt.plot(sim['detector_cubesat_group.thrust_cp'][0, :])
        plt.plot(sim['detector_cubesat_group.thrust_cp'][1, :])
        plt.plot(sim['detector_cubesat_group.thrust_cp'][2, :])
        plt.show()

        plt.title(
            'thrust profile (high during\nobservation phase after optimization)'
        )
        plt.plot(sim['observation_phase_indicator'])
        plt.plot(sim['optics_cubesat_group.thrust'][0, :])
        plt.plot(sim['optics_cubesat_group.thrust'][1, :])
        plt.plot(sim['optics_cubesat_group.thrust'][2, :])
        plt.plot(sim['detector_cubesat_group.thrust'][0, :])
        plt.plot(sim['detector_cubesat_group.thrust'][1, :])
        plt.plot(sim['detector_cubesat_group.thrust'][2, :])
        plt.show()

        plt.title(
            'sun_LOS; observation phase is within phase where sun_LOS==1')
        plt.plot(sim['observation_phase_indicator'])
        plt.plot(sim['optics_cubesat_group.sun_LOS'][0, :])
        plt.plot(sim['detector_cubesat_group.sun_LOS'][0, :])
        plt.show()

        plt.title(
            'sun_LOS, percent_exposed_area;\n percent_exposed_area zero when sun_LOS == 0'
        )
        plt.plot(sim['optics_cubesat_group.sun_LOS'][0, :])
        plt.plot(sim['detector_cubesat_group.sun_LOS'][0, :])
        plt.plot(sim['optics_cubesat_group.percent_exposed_area'][0, :])
        plt.plot(sim['detector_cubesat_group.percent_exposed_area'][0, :])
        plt.show()

        # these are zero when Euler angles are zero
        plt.title('body_rates')
        plt.plot(sim['optics_cubesat_group.body_rates'][0, :])
        plt.plot(sim['optics_cubesat_group.body_rates'][1, :])
        plt.plot(sim['optics_cubesat_group.body_rates'][2, :])
        plt.show()

        plt.title('rw_speed must be smooth')
        plt.plot(sim['optics_cubesat_group.rw_speed'][0, :])
        plt.plot(sim['optics_cubesat_group.rw_speed'][1, :])
        plt.plot(sim['optics_cubesat_group.rw_speed'][2, :])
        plt.show()

        # torques counteract each other when Euler angles are zero
        plt.title('rw_torque')
        plt.plot(sim['optics_cubesat_group.rw_torque'][0, :])
        plt.plot(sim['optics_cubesat_group.rw_torque'][1, :])
        plt.plot(sim['optics_cubesat_group.rw_torque'][2, :])
        plt.show()

        plt.title('body_torque')
        plt.plot(sim['optics_cubesat_group.body_torque'][0, :])
        plt.plot(sim['optics_cubesat_group.body_torque'][1, :])
        plt.plot(sim['optics_cubesat_group.body_torque'][2, :])
        plt.show()

        plt.title('rw_torque - body_torque')
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

        plt.title('SOC should NOT be constant')
        plt.plot(sim['optics_cubesat_group.soc'].flatten())
        plt.show()

        plt.title('should be 2.4180232931306733 when soc << 0')
        plt.plot(sim['optics_cubesat_group.voltage'].flatten())
        plt.plot(3 + (np.exp(sim['optics_cubesat_group.soc'].flatten()) - 1) /
                 (np.exp(1) - 1))
        plt.show()

        plt.title(
            'battery power, power draw optics s/c (differ by const factor)')
        plt.plot(sim['optics_cubesat_group.power'].flatten())
        plt.plot(sim['detector_cubesat_group.power'].flatten())
        plt.show()

        plt.title(
            'battery power, power draw detector s/c (differ by const factor)')
        plt.plot(sim['optics_cubesat_group.battery_power'])
        plt.plot(sim['detector_cubesat_group.battery_power'])
        plt.show()

        # nonnegative
        plt.title('separation_m')
        plt.plot(sim['separation_m'].flatten())
        plt.show()

        plt.title(
            'separation_error_during_observation_mm 0 outside observation phase'
        )
        plt.plot(sim['observation_phase_indicator'])
        plt.plot(sim['separation_error_during_observation_mm'].flatten())
        plt.show()

        plt.title('telescope_cos_view_angle < 1')
        plt.plot(sim['telescope_cos_view_angle'].flatten())
        plt.show()

        # # between 0 and 1
        # plt.plot(sim['optics_cos_view_angle_error'].flatten())
        # plt.plot(sim['optics_cos_view_angle_error_during_observation'].flatten())
        # plt.show()

        # # arcsec
        # plt.plot(sim['optics_view_angle_error'].flatten())
        # plt.show()
        # plt.plot(sim['detector_view_angle_error'].flatten())
        # plt.show()
