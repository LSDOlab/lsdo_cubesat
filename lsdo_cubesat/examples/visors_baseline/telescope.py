from csdl import Model
import csdl
import numpy as np

# from lsdo_cubesat.cubesat_group import Cubesat
from lsdo_cubesat.telescope.telescope_configuration import TelescopeConfiguration
from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.examples.visors_baseline.cubesat_group import Cubesat
from lsdo_cubesat.operations.sun_direction import SunDirection


class Telescope(Model):

    def initialize(self):
        self.parameters.declare('swarm', types=SwarmParams)
        self.parameters.declare('comm', types=bool, default=False)
        self.parameters.declare('duration', types=float)

    def define(self):
        swarm = self.parameters['swarm']
        comm = self.parameters['comm']

        num_times = swarm['num_times']
        num_cp = swarm['num_cp']
        step_size = swarm['step_size']
        duration = swarm['duration']

        earth_orbit_angular_speed_rad_min = 2 * np.pi / 365 * 1 / 24 * 1 / 60
        step_size_min = duration / num_times
        earth_orbit_angular_position = earth_orbit_angular_speed_rad_min * step_size_min * np.arange(
            num_times)
        v = np.zeros((3, num_times))
        v[0, :] = np.cos(earth_orbit_angular_position)
        v[1, :] = np.sin(earth_orbit_angular_position)
        sun_direction = self.create_input(
            'sun_direction',
            shape=(3, num_times),
            val=v,
        )
        for cubesat in swarm.children:
            cubesat_name = cubesat['name']
            submodel_name = '{}_cubesat'.format(cubesat_name)
            self.add(
                Cubesat(
                    num_times=num_times,
                    num_cp=num_cp,
                    step_size=step_size,
                    cubesat=cubesat,
                    # mtx=get_bspline_mtx(num_cp, num_times, order=4),
                ),
                name='{}_cubesat'.format(cubesat_name),
                promotes=['reference_orbit_state_km'],
            )
            self.connect('sun_direction',
                         '{}.sun_direction'.format(submodel_name))

        self.add(
            TelescopeConfiguration(swarm=swarm),
            name='telescope_config',
        )
        # self.connect(
        #     'optics_cubesat.sun_LOS',
        #     'optics_sun_LOS',
        # )
        # self.connect(
        #     'detector_cubesat.sun_LOS',
        #     'detector_sun_LOS',
        # )
        self.connect(
            'optics_cubesat.relative_orbit_state_m',
            'optics_relative_orbit_state_m',
        )
        self.connect(
            'detector_cubesat.relative_orbit_state_m',
            'detector_relative_orbit_state_m',
        )
        self.connect(
            'optics_cubesat.orbit_state_km',
            'optics_orbit_state_km',
        )
        self.connect(
            'detector_cubesat.orbit_state_km',
            'detector_orbit_state_km',
        )
        # self.connect(
        #     'optics_cubesat.B_from_ECI',
        #     'optics_B_from_ECI',
        # )
        # self.connect(
        #     'detector_cubesat.B_from_ECI',
        #     'detector_B_from_ECI',
        # )
        # self.connect(
        #     'optics_cubesat.sun_pointing_constraint',
        #     'optics_sun_pointing_constraint',
        # )
        # self.connect(
        #     'detector_cubesat.sun_pointing_constraint',
        #     'detector_sun_pointing_constraint',
        # )

        optics_total_propellant_used = self.declare_variable(
            'optics_total_propellant_used')
        detector_total_propellant_used = self.declare_variable(
            'detector_total_propellant_used')

        total_propellant_used = optics_total_propellant_used + detector_total_propellant_used
        self.register_output('total_propellant_used', total_propellant_used)

        self.connect(
            'optics_cubesat.total_propellant_used',
            'optics_total_propellant_used',
        )
        self.connect(
            'detector_cubesat.total_propellant_used',
            'detector_total_propellant_used',
        )
        # if comm is True:
        #     optics_cubesat_group_total_data = self.declare_variable(
        #         'optics_cubesat_group_total_data')
        #     detector_cubesat_group_total_data = self.declare_variable(
        #         'detector_cubesat_group_total_data')

        # # FOR FUTURE DEVELOPERS:
        # # THIS USED TO BE HERE FOR THE COMMUNCATION DISCIPLINES
        # # IT WASN'T CHECKED TO MAKE SURE EVERYTHING WORKS WHEN comm is True
        # # total_data_downloaded = sunshade_cubesat_group_total_Data + optics_cubesat_group_total_Data + detector_cubesat_group_total_Data
        # # total_data_downloaded = optics_cubesat_group_total_data + detector_cubesat_group_total_data
        # # +5.e-14*ks_masked_distance_sunshade_optics_km
        # # +5.e-14 *ks_masked_distance_optics_detector_km
        # # self.register_output('total_data_downloaded', total_data_downloaded)

        # # for cubesat in swarm.children:
        # #     name = cubesat['name']

        # #     # self.connect(
        # #     #     '{}_cubesat_group.position_km'.format(name),
        # #     #     '{}_cubesat_group_position_km'.format(name),
        # #     # )

        # #     self.connect(
        # #         '{}_cubesat_group.total_propellant_used'.format(name),
        # #         '{}_cubesat_group_total_propellant_used'.format(name),
        # #     )

        # #     if comm is True:
        # #         self.connect(
        # #             '{}_cubesat_group.total_data'.format(name),
        # #             '{}_cubesat_group_total_data'.format(name),
        # #         )

        # Objective with regularizaiton term
        optics_relative_orbit_state_m = self.declare_variable(
            'optics_relative_orbit_state_m', shape=(6, num_times))
        detector_relative_orbit_state_m = self.declare_variable(
            'detector_relative_orbit_state_m', shape=(6, num_times))
        x = csdl.sum(optics_relative_orbit_state_m[:3, :]**2, axes=(0, ))
        y = csdl.sum(detector_relative_orbit_state_m[:3, :]**2, axes=(0, ))

        separation_error_during_observation = self.declare_variable(
            'separation_error_during_observation',
            shape=num_times,
        )
        view_plane_error_during_observation = self.declare_variable(
            'view_plane_error_during_observation', shape=num_times)
        # obj_comp = ExecComp(
        #     # 'obj= 0.01 * total_propellant_used- 1e-5 * total_data_downloaded + 1e-4 * (0'
        #     'obj= 0.01 * total_propellant_used + 1e-4 * (0'
        #     '+ masked_normal_distance_optics_detector_mm_sq_sum'
        #     '+ masked_distance_optics_detector_mm_sq_sum)/{}'
        #     '+ 1e-3 * ('
        #     '+ optics_cubesat_group_relative_orbit_state_sq_sum'
        #     '+ detector_cubesat_group_relative_orbit_state_sq_sum'
        #     ') / {}'.format(num_times, num_times))

        # TODO: get reasonable coefficients for regularization term
        obj = 0.01 * total_propellant_used + 1e2 * (
            csdl.sum((1e3 * view_plane_error_during_observation)**2 +
                     (1e3 * separation_error_during_observation)**2) +
            1e-3 * csdl.sum(x**2 + y**2)) / num_times

        # propellant on the order of kg
        # relative distance to reference orbit on the order of m
        self.register_output('obj', obj)
