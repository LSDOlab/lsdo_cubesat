import numpy as np
from openmdao.api import Group, IndepVarComp

from lsdo_cubesat.attitude.attitude_rk4_comp import AttitudeRK4Comp
from lsdo_cubesat.attitude.attitude_rk4_gravity_comp import AttitudeRK4GravityComp
from lsdo_cubesat.attitude.attitude_state_decomposition_comp import \
    AttitudeStateDecompositionComp
from lsdo_cubesat.attitude.inertia_ratios_comp import InertiaRatiosComp
from lsdo_cubesat.attitude.rot_mtx_to_rpy import RotMtxToRollPitchYaw
from lsdo_cubesat.utils.finite_difference_comp import FiniteDifferenceComp
from lsdo_cubesat.utils.normalize_last_quaternion import \
    NormalizeLastQuaternion
from lsdo_cubesat.utils.quaternion_to_rot_mtx import QuaternionToRotMtx
from lsdo_cubesat.utils.api import (ArrayExpansionComp, ArrayReorderComp,
                                    BsplineComp, PowerCombinationComp)
from lsdo_cubesat.utils.api import get_bspline_mtx


class AttitudeOdeGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('num_cp', types=int)
        self.options.declare('cubesat')
        self.options.declare('mtx')

    def setup(self):
        num_times = self.options['num_times']
        step_size = self.options['step_size']
        num_cp = self.options['num_cp']
        cubesat = self.options['cubesat']
        mtx = self.options['mtx']

        # CADRE mass props (3U)
        I = np.array([18.0, 18.0, 6.0]) * 1e-3

        # Initial angular velocity and quaternion
        # wq0 = np.array([-1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
        wq0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        comp = IndepVarComp()
        comp.add_output('times',
                        units='s',
                        val=np.linspace(0., step_size * (num_times - 1),
                                        num_times))
        comp.add_output('external_torques_x_cp', val=np.zeros(num_cp))
        comp.add_output('external_torques_y_cp', val=np.zeros(num_cp))
        comp.add_output('external_torques_z_cp', val=np.zeros(num_cp))
        comp.add_output('initial_angular_velocity_orientation', val=wq0)
        comp.add_output('mass_moment_inertia_b_frame_km_m2', val=I)
        comp.add_design_var('external_torques_x_cp')
        comp.add_design_var('external_torques_y_cp')
        comp.add_design_var('external_torques_z_cp')
        self.add_subsystem('inputs_comp', comp, promotes=['*'])

        # Expand external_torques
        for var_name in [
                'external_torques_x',
                'external_torques_y',
                'external_torques_z',
        ]:
            comp = BsplineComp(
                num_pt=num_times,
                num_cp=num_cp,
                jac=get_bspline_mtx(num_cp, num_times),
                in_name='{}_cp'.format(var_name),
                out_name=var_name,
            )
            self.add_subsystem('{}_comp'.format(var_name),
                               comp,
                               promotes=['*'])

        # Integrate attitude dynamics
        self.add_subsystem(
            'attitude_rk4',
            AttitudeRK4GravityComp(
                num_times=num_times,
                step_size=step_size,
                moment_inertia_ratios=np.array([2.0 / 3.0, -2.0 / 3.0, 0]),
            ),
            promotes=['*'],
        )

        # Decompose angular velocity and orientation
        self.add_subsystem(
            'attitude_state_decomp',
            AttitudeStateDecompositionComp(
                num_times=num_times,
                angular_velocity_orientation='angular_velocity_orientation',
                angular_velocity_name='angular_velocity',
                quaternion_name='quaternions'),
            promotes=['*'],
        )

        # Integrator normalizes all but last quaternion
        self.add_subsystem(
            'normalize_last_quaternion',
            NormalizeLastQuaternion(num_times=num_times, ),
            promotes=['*'],
        )

        # Compute rotation matrix
        self.add_subsystem('rot_mtx_b_i_3x3xn_comp',
                           QuaternionToRotMtx(num_times=num_times),
                           promotes=['*'])

        # Compute roll, pitch, yaw from rotation matrix (to use in finite
        # difference for roll and pitch rate constraints)
        self.add_subsystem(
            'rot_mtx_to_rpy',
            RotMtxToRollPitchYaw(
                mtx_name='rot_mtx_b_i_3x3xn',
                num_times=num_times,
            ),
            promotes=['*'],
        )

        # Transpose rotation matrix
        comp = ArrayReorderComp(
            in_shape=(3, 3, num_times),
            out_shape=(3, 3, num_times),
            in_subscripts='ijn',
            out_subscripts='jin',
            in_name='rot_mtx_b_i_3x3xn',
            out_name='rot_mtx_i_b_3x3xn_fast',
        )
        self.add_subsystem('rot_mtx_i_b_3x3xn_comp', comp, promotes=['*'])

        # Get finite difference of roll, pitch, yaw rates to set constraints
        for var_name in [
                'times',
                'roll',
                'pitch',
                'yaw',
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

        # Set roll, pitch, yaw rate constraints
        for var_name in [
                'roll',
                'pitch',
                'yaw',
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
