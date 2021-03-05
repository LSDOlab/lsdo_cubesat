import numpy as np
from openmdao.api import ExecComp
from omtools.api import Group

from lsdo_cubesat.api import Swarm
from lsdo_cubesat.alignment.alignment_group import AlignmentGroup
from lsdo_cubesat.cubesat_group import CubesatGroup
from lsdo_cubesat.orbit.reference_orbit_group import ReferenceOrbitGroup
from lsdo_cubesat.solar.smt_exposure import smt_exposure
from lsdo_cubesat.utils.api import get_bspline_mtx
from lsdo_cubesat.utils.comps.arithmetic_comps.elementwise_max_comp import \
    ElementwiseMaxComp
from lsdo_cubesat.examples.data.cubesat_xdata import cubesat_xdata as az
from lsdo_cubesat.examples.data.cubesat_ydata import cubesat_ydata as el
from lsdo_cubesat.examples.data.cubesat_zdata import cubesat_zdata as yt


class SwarmGroup(Group):
    def initialize(self):
        self.options.declare('swarm', types=Swarm)
        self.options.declare('add_battery', types=bool)
        self.options.declare('optimize_plant', types=bool)
        self.options.declare('attitude_integrator', types=bool)
        self.options.declare('battery_time_scale', types=float)
        self.options.declare('attitude_time_scale', types=float)

    def setup(self):
        swarm = self.options['swarm']

        num_times = swarm['num_times']
        num_cp = swarm['num_cp']
        step_size = swarm['step_size']
        add_battery = self.options['add_battery']
        mtx = get_bspline_mtx(num_cp, num_times, order=4)
        optimize_plant = self.options['optimize_plant']
        attitude_integrator = self.options['attitude_integrator']
        battery_time_scale = self.options['battery_time_scale']
        attitude_time_scale = self.options['attitude_time_scale']

        self.add_subsystem(
            'reference_orbit_group',
            ReferenceOrbitGroup(
                num_times=num_times,
                step_size=step_size,
                cubesat=swarm.children[0],
            ),
            promotes=['*'],
        )

        sm = None
        if add_battery:
            # load training data
            # az = np.genfromtxt(here + '/solar/training/data/cubesat_xdata.csv',
            #                    delimiter=',')
            # el = np.genfromtxt(here + '/solar/training/data/cubesat_ydata.csv',
            #                    delimiter=',')
            # yt = np.genfromtxt(here + '/solar/training/data/cubesat_zdata.csv',
            #                    delimiter=',')

            # generate surrogate model with 20 training points
            # must be the same as the number of points used to create model
            sm = smt_exposure(20, az, el, yt)

        # for cubesat in swarm.children:
        #     name = cubesat['name']
        #     for ground_station in cubesat.children:
        #         group = CubesatGroup(
        #             num_times=num_times,
        #             num_cp=num_cp,
        #             step_size=step_size,
        #             cubesat=cubesat,
        #             mtx=mtx,
        #             ground_station=ground_station,
        #             add_battery=add_battery,
        #             sm=sm,
        #             optimize_plant=optimize_plant,
        #             attitude_integrator=attitude_integrator,
        #             attitude_time_scale=attitude_time_scale,
        #             battery_time_scale=battery_time_scale,
        #         )
        #     self.add_subsystem(
        #         '{}_cubesat_group'.format(name),
        #         group,
        #         promotes=['*'],
        #     )
        self.add_subsystem(
            'alignment_group',
            AlignmentGroup(
                swarm=swarm,
                mtx=mtx,
            ),
            promotes=['*'],
        )

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
                '{}_cubesat_group.total_data'.format(name),
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
            for ground_station in cubesat.children:
                Ground_station_name = ground_station['name']

                for var_name in ['orbit_state_km', 'rot_mtx_i_b_3x3xn']:
                    self.connect(
                        '{}_cubesat_group.{}'.format(cubesat_name, var_name),
                        '{}_cubesat_group.{}_comm_group.{}'.format(
                            cubesat_name, Ground_station_name, var_name))

                # self.connect(
                #     'times', '{}_cubesat_group.attitude_group.times'.format(
                #         cubesat_name))
