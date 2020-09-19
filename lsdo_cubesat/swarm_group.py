import numpy as np
from openmdao.api import Group, ExecComp

from lsdo_utils.api import get_bspline_mtx
from lsdo_utils.comps.arithmetic_comps.elementwise_max_comp import ElementwiseMaxComp

from lsdo_cubesat.cubesat_group import CubesatGroup
from lsdo_cubesat.alignment.alignment_group import AlignmentGroup
from lsdo_cubesat.orbit.reference_orbit_group import ReferenceOrbitGroup
from lsdo_cubesat.communication.ground_station import Ground_station
from lsdo_cubesat.solar.smt_exposure import smt_exposure


class SwarmGroup(Group):
    def initialize(self):
        self.options.declare('swarm')
        self.options.declare('add_battery', types=bool)
        self.options.declare('optimize_plant')

    def setup(self):
        swarm = self.options['swarm']

        num_times = swarm['num_times']
        num_cp = swarm['num_cp']
        step_size = swarm['step_size']
        add_battery = self.options['add_battery']
        mtx = get_bspline_mtx(num_cp, num_times, order=4)
        optimize_plant = self.options['optimize_plant']

        group = ReferenceOrbitGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=swarm.children[0],
        )
        self.add_subsystem('reference_orbit_group', group, promotes=['*'])

        sm = None
        if add_battery:
            # load training data
            az = np.genfromtxt(
                '/Users/victor/Desktop/lsdo_cubesat/lsdo_cubesat/training_data/arrow_xData.csv',
                delimiter=',')
            el = np.genfromtxt(
                '/Users/victor/Desktop/lsdo_cubesat/lsdo_cubesat/training_data/arrow_yData.csv',
                delimiter=',')
            yt = np.genfromtxt(
                '/Users/victor/Desktop/lsdo_cubesat/lsdo_cubesat/training_data/arrow_zData.csv',
                delimiter=',')

            # generate surrogate model with 20 training points
            # must be the same as the number of points used to create model
            sm = smt_exposure(20, az, el, yt)

        for cubesat in swarm.children:
            name = cubesat['name']
            for Ground_station in cubesat.children:
                group = CubesatGroup(
                    num_times=num_times,
                    num_cp=num_cp,
                    step_size=step_size,
                    cubesat=cubesat,
                    mtx=mtx,
                    Ground_station=Ground_station,
                    add_battery=add_battery,
                    sm=sm,
                    optimize_plant=optimize_plant,
                )
            self.add_subsystem('{}_cubesat_group'.format(name), group)

        group = AlignmentGroup(
            swarm=swarm,
            mtx=mtx,
        )
        self.add_subsystem('alignment_group', group, promotes=['*'])

        comp = ExecComp(
            'total_propellant_used' +
            '=sunshade_cubesat_group_total_propellant_used' +
            '+optics_cubesat_group_total_propellant_used' +
            '+detector_cubesat_group_total_propellant_used'
            # '+5.e-14*ks_masked_distance_sunshade_optics_km' +
            # '+5.e-14 *ks_masked_distance_optics_detector_km'
        )
        self.add_subsystem('total_propellant_used_comp', comp, promotes=['*'])

        comp = ExecComp(
            'total_data_downloaded' + '=sunshade_cubesat_group_total_Data' +
            '+optics_cubesat_group_total_Data' +
            '+detector_cubesat_group_total_Data'
            # '+5.e-14*ks_masked_distance_sunshade_optics_km' +
            # '+5.e-14 *ks_masked_distance_optics_detector_km'
        )
        self.add_subsystem('total_data_downloaded_comp', comp, promotes=['*'])

        for cubesat in swarm.children:
            name = cubesat['name']

            self.connect(
                '{}_cubesat_group.position_km'.format(name),
                '{}_cubesat_group_position_km'.format(name),
            )

            self.connect(
                '{}_cubesat_group.total_propellant_used'.format(name),
                '{}_cubesat_group_total_propellant_used'.format(name),
            )

            self.connect(
                '{}_cubesat_group.total_Data'.format(name),
                '{}_cubesat_group_total_Data'.format(name),
            )

            for var_name in [
                    'radius',
                    'reference_orbit_state',
            ]:
                self.connect(
                    var_name,
                    '{}_cubesat_group.{}'.format(name, var_name),
                )

        for cubesat in swarm.children:
            cubesat_name = cubesat['name']
            for Ground_station in cubesat.children:
                Ground_station_name = Ground_station['name']

                for var_name in ['orbit_state_km', 'rot_mtx_i_b_3x3xn']:
                    self.connect(
                        '{}_cubesat_group.{}'.format(cubesat_name, var_name),
                        '{}_cubesat_group.{}_comm_group.{}'.format(
                            cubesat_name, Ground_station_name, var_name))

                # self.connect(
                #     'times', '{}_cubesat_group.attitude_group.times'.format(
                #         cubesat_name))
