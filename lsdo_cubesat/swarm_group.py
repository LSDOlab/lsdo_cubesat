from csdl import Model

# from lsdo_cubesat.cubesat_group import Cubesat
from lsdo_cubesat.alignment.alignment_group import Alignment
from lsdo_cubesat.orbit.reference_orbit_group import ReferenceOrbit
from lsdo_cubesat.communication.ground_station import GroundStationParams
from lsdo_cubesat.swarm.swarm import SwarmParams
from lsdo_cubesat.cubesat_group import Cubesat


class Swarm(Model):
    def initialize(self):
        self.parameters.declare('swarm', types=SwarmParams)

    def define(self):
        swarm = self.parameters['swarm']

        num_times = swarm['num_times']
        num_cp = swarm['num_cp']
        step_size = swarm['step_size']

        self.add(
            ReferenceOrbit(
                num_times=num_times,
                step_size=step_size,
            ),
            name='reference_orbit',
        )

        for cubesat in swarm.children:
            name = cubesat['name']
            self.add(
                Cubesat(
                    num_times=num_times,
                    num_cp=num_cp,
                    step_size=step_size,
                    cubesat=cubesat,
                    # mtx=get_bspline_mtx(num_cp, num_times, order=4),
                ),
                name='{}_cubesat_group'.format(name),
                promotes=[],
            )

        self.add(
            Alignment(swarm=swarm),
            name='alignment',
        )

        sunshade_cubesat_group_total_Data = self.declare_variable(
            'sunshade_cubesat_group_total_data')
        optics_cubesat_group_total_data = self.declare_variable(
            'optics_cubesat_group_total_data')
        detector_cubesat_group_total_data = self.declare_variable(
            'detector_cubesat_group_total_data')

        # total_data_downloaded = sunshade_cubesat_group_total_Data + optics_cubesat_group_total_Data + detector_cubesat_group_total_Data
        # '+5.e-14*ks_masked_distance_sunshade_optics_km' +
        # '+5.e-14 *ks_masked_distance_optics_detector_km'

        # sunshade_cubesat_group_total_propellant_used = self.declare_variable(
        #     'sunshade_cubesat_group_total_propellant_used')
        optics_cubesat_group_total_propellant_used = self.declare_variable(
            'optics_cubesat_group_total_propellant_used')
        detector_cubesat_group_total_propellant_used = self.declare_variable(
            'detector_cubesat_group_total_propellant_used')
        ks_masked_distance_sunshade_optics_km = self.declare_variable(
            'ks_masked_distance_sunshade_optics_km')
        ks_masked_distance_optics_detector_km = self.declare_variable(
            'ks_masked_distance_optics_detector_km')

        # total_propellant_used = sunshade_cubesat_group_total_propellant_used + optics_cubesat_group_total_propellant_used + detector_cubesat_group_total_propellant_used
        total_propellant_used = optics_cubesat_group_total_propellant_used + detector_cubesat_group_total_propellant_used
        # +5.e-14*ks_masked_distance_sunshade_optics_km
        # +5.e-14 *ks_masked_distance_optics_detector_km
        self.register_output('total_propellant_used', total_propellant_used)

        # sunshade_cubesat_group_total_Data = self.declare_variable(
        #     'sunshade_cubesat_group_total_Data')
        optics_cubesat_group_total_data = self.declare_variable(
            'optics_cubesat_group_total_data')
        detector_cubesat_group_total_data = self.declare_variable(
            'detector_cubesat_group_total_data')
        # total_data_downloaded = sunshade_cubesat_group_total_Data + optics_cubesat_group_total_Data + detector_cubesat_group_total_Data
        total_data_downloaded = optics_cubesat_group_total_data + detector_cubesat_group_total_data
        # +5.e-14*ks_masked_distance_sunshade_optics_km
        # +5.e-14 *ks_masked_distance_optics_detector_km
        self.register_output('total_data_downloaded', total_data_downloaded)

        for cubesat in swarm.children:
            name = cubesat['name']

            # self.connect(
            #     '{}_cubesat_group.position_km'.format(name),
            #     '{}_cubesat_group_position_km'.format(name),
            # )

            self.connect(
                '{}_cubesat_group.total_propellant_used'.format(name),
                '{}_cubesat_group_total_propellant_used'.format(name),
            )

            self.connect(
                '{}_cubesat_group.total_data'.format(name),
                '{}_cubesat_group_total_data'.format(name),
            )

            # for var_name in [
            #         'radius',
            #         'reference_orbit_state',
            # ]:
            #     self.connect(
            #         var_name,
            #         '{}_cubesat_group.{}'.format(name, var_name),
            #     )

        for cubesat in swarm.children:
            cubesat_name = cubesat['name']
            # for ground_station in cubesat.children:
            #     ground_station_name = ground_station['name']

            #     for var_name in ['orbit_state_km', 'rot_mtx_i_b_3x3xn']:
            #         self.connect(
            #             '{}_cubesat_group.{}'.format(cubesat_name, var_name),
            #             '{}_cubesat_group.{}_comm_group.{}'.format(
            #                 cubesat_name, ground_station_name, var_name))

            #     # self.connect(
            #     #     'times', '{}_cubesat_group.attitude_group.times'.format(
            #     #         cubesat_name))
            # self.connect(
            #     '{}_cubesat_group.relative_orbit_state_sq_sum'.format(
            #         cubesat_name),
            #     '{}_cubesat_group_relative_orbit_state_sq_sum'.format(
            #         cubesat_name),
            # )
        # TODO: define these variables
        # obj = 0.01 * total_propellant_used - 1e-5 * total_data_downloaded + 1e-4 * (
        #     0 + masked_normal_distance_sunshade_detector_mm_sq_sum +
        #     masked_normal_distance_optics_detector_mm_sq_sum +
        #     masked_distance_sunshade_optics_mm_sq_sum +
        #     masked_distance_optics_detector_mm_sq_sum) / num_times + 1e-3 * (
        #         sunshade_cubesat_group_relative_orbit_state_sq_sum +
        #         optics_cubesat_group_relative_orbit_state_sq_sum +
        #         detector_cubesat_group_relative_orbit_state_sq_sum) / num_times
        # self.register_output('obj', obj)
        # self.add_objective('obj', scaler=1.e-3)
