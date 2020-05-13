import numpy as np

from openmdao.api import Group, IndepVarComp

from lsdo_cubesat.attitude.attitude_group import AttitudeGroup
from lsdo_cubesat.propulsion.propulsion_group import PropulsionGroup
from lsdo_cubesat.aerodynamics.aerodynamics_group import AerodynamicsGroup
from lsdo_cubesat.orbit.orbit_group import OrbitGroup
from lsdo_cubesat.communication.communication_group import CommGroup
from lsdo_cubesat.mean_motion_group import MeanMotionGroup


class CubesatGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')
        self.options.declare('mtx')

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']

        times = np.linspace(0., step_size * (num_times - 1), num_times)

        comp = IndepVarComp()
        comp.add_output('times', val=times)
        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        group = AttitudeGroup(
            num_times=num_times,
            step_size=step_size,
            num_cp=num_cp,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('attitude_group', group, promotes=['*'])

        group = PropulsionGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('propulsion_group', group, promotes=['*'])

        group = AerodynamicsGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('aerodynamics_group', group, promotes=['*'])

        group = OrbitGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('orbit_group', group, promotes=['*'])

        self.add_subsystem('mean_motion_group',
                           MeanMotionGroup(num_times=num_times, ),
                           promotes=['*'])

        group = CommGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=cubesat,
            mtx=mtx,
        )
        self.add_subsystem('commnication_group', group, promotes=['*'])
        # self.connect('rot_mtx_i_b_3x3xn_comp',[])
