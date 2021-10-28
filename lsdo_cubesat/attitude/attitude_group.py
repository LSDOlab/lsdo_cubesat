import csdl
import numpy as np

from csdl import Model
import csdl
from csdl.utils.get_bspline_mtx import get_bspline_mtx

from lsdo_cubesat.utils.finite_difference_comp import FiniteDifferenceComp
from lsdo_cubesat.attitude.rot_mtx_b_i_comp import RotMtxBIComp


class AttitudeGroup(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']

        # self.create_input('roll_cp', val=2. * np.pi * np.random.rand(num_cp))
        # self.create_input('pitch_cp', val=2. * np.pi * np.random.rand(num_cp))
        roll_cp = self.create_input('roll_cp', val=np.ones(num_cp))
        pitch_cp = self.create_input('pitch_cp', val=np.ones(num_cp))
        self.add_design_variable('roll_cp')
        self.add_design_variable('pitch_cp')

        roll = csdl.matvec(get_bspline_mtx(num_cp, num_times), roll_cp)
        pitch = csdl.matvec(get_bspline_mtx(num_cp, num_times), pitch_cp)
        self.register_output('roll', roll)
        self.register_output('pitch', pitch)

        rot_mtx_b_i_3x3xn = csdl.custom(
            roll,
            pitch,
            op=RotMtxBIComp(num_times=num_times),
        )
        rot_mtx_i_b_3x3xn = csdl.reorder_axes(rot_mtx_b_i_3x3xn, 'ijn->jin')
        self.register_output('rot_mtx_i_b_3x3xn', rot_mtx_i_b_3x3xn)

        # for var_name, var in [
        #         # ('times', x),
        #     ('roll', roll),
        #     ('pitch', pitch),
        # ]:
        #     self.register_output(
        #         'd{}'.format(var_name),
        #         csdl.custom(
        #             var,
        #             op=FiniteDifferenceComp(
        #                 num_times=num_times,
        #                 in_name=var_name,
        #                 out_name='d{}'.format(var_name),
        #             ),
        #         ),
        #     )

        rad_deg = np.pi / 180.

        # for var_name in [
        #         'roll',
        #         'pitch',
        # ]:
        #     comp = PowerCombinationComp(shape=(num_times, ),
        #                                 out_name='{}_rate'.format(var_name),
        #                                 powers_dict={
        #                                     'd{}'.format(var_name): 1.,
        #                                     'dtimes': -1.,
        #                                 })
        #     comp.add_constraint('{}_rate'.format(var_name),
        #                         lower=-10. * rad_deg,
        #                         upper=10. * rad_deg,
        #                         linear=True)
        #     self.add('{}_rate_comp'.format(var_name), comp, promotes=['*'])
