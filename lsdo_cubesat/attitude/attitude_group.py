import numpy as np

from openmdao.api import Group, IndepVarComp

from lsdo_cubesat.utils.api import ArrayReorderComp, BsplineComp, PowerCombinationComp

from lsdo_cubesat.utils.finite_difference_comp import FiniteDifferenceComp
from lsdo_cubesat.attitude.rot_mtx_b_i_comp import RotMtxBIComp


class AttitudeGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('cubesat')
        self.options.declare('mtx')
        self.options.declare('step_size', types=float)

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        mtx = self.options['mtx']
        step_size = self.options['step_size']

        comp = IndepVarComp()
        comp.add_output('times',
                        units='s',
                        val=np.linspace(0., step_size * (num_times - 1),
                                        num_times))
        # comp.add_output('roll_cp', val=2. * np.pi * np.random.rand(num_cp))
        # comp.add_output('pitch_cp', val=2. * np.pi * np.random.rand(num_cp))
        comp.add_output('roll_cp', val=np.ones(num_cp))
        comp.add_output('pitch_cp', val=np.ones(num_cp))
        comp.add_design_var('roll_cp')
        comp.add_design_var('pitch_cp')
        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        for var_name in ['roll', 'pitch']:
            comp = BsplineComp(
                num_pt=num_times,
                num_cp=num_cp,
                jac=mtx,
                in_name='{}_cp'.format(var_name),
                out_name=var_name,
            )
            self.add_subsystem('{}_comp'.format(var_name),
                               comp,
                               promotes=['*'])

        comp = RotMtxBIComp(num_times=num_times)
        self.add_subsystem('rot_mtx_b_i_3x3xn_comp', comp, promotes=['*'])

        comp = ArrayReorderComp(
            in_shape=(3, 3, num_times),
            out_shape=(3, 3, num_times),
            in_subscripts='ijn',
            out_subscripts='jin',
            in_name='rot_mtx_b_i_3x3xn',
            out_name='rot_mtx_i_b_3x3xn',
        )
        self.add_subsystem('rot_mtx_i_b_3x3xn_comp', comp, promotes=['*'])

        for var_name in [
                'times',
                'roll',
                'pitch',
        ]:
            comp = FiniteDifferenceComp(
                num_times=num_times,
                in_name=var_name,
                out_name='d{}'.format(var_name),
            )
            self.add_subsystem('d{}_comp'.format(var_name),
                               comp,
                               promotes=['*'])

        rad_deg = np.pi / 180.

        for var_name in [
                'roll',
                'pitch',
        ]:
            comp = PowerCombinationComp(shape=(num_times, ),
                                        out_name='{}_rate'.format(var_name),
                                        powers_dict={
                                            'd{}'.format(var_name): 1.,
                                            'dtimes': -1.,
                                        })
            comp.add_constraint('{}_rate'.format(var_name),
                                lower=-10. * rad_deg,
                                upper=10. * rad_deg,
                                linear=True)
            self.add_subsystem('{}_rate_comp'.format(var_name),
                               comp,
                               promotes=['*'])
