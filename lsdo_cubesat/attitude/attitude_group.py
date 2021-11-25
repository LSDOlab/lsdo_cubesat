import csdl
import numpy as np

from csdl import Model
import csdl
from csdl.utils.get_bspline_mtx import get_bspline_mtx

from lsdo_cubesat.utils.finite_difference_comp import FiniteDifferenceComp
from lsdo_cubesat.attitude.attitude_rk4_gravity_2 import AttitudeRK4GravityComp
from lsdo_cubesat.operations.rw_speed_rk4 import RWSpeedRK4


class AttitudeGroup(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('max_torque', default=0.007)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        max_torque = self.parameters['max_torque']

        external_torques_x = self.create_input(
            'external_torques_x',
            val=0,
            shape=num_times,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        external_torques_y = self.create_input(
            'external_torques_y',
            val=0,
            shape=num_times,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        external_torques_z = self.create_input(
            'external_torques_z',
            val=0,
            shape=num_times,
            desc=
            'External torques applied to spacecraft, e.g. ctrl inputs, drag')

        self.add_design_variable('external_torques_x',
                                 lower=-max_torque,
                                 upper=max_torque)
        self.add_design_variable('external_torques_y',
                                 lower=-max_torque,
                                 upper=max_torque)
        self.add_design_variable('external_torques_z',
                                 lower=-max_torque,
                                 upper=max_torque)
        initial_rw_speed = self.declare_variable('initial_rw_speed',
                                                 shape=(3, ),
                                                 val=0)
        rw_speed = csdl.custom(external_torques_x,
                               external_torques_y,
                               external_torques_z,
                               initial_rw_speed,
                               op=RWSpeedRK4(
                                   num_times=num_times,
                                   step_size=step_size,
                               ))
        self.register_output('rw_speed', rw_speed)

        # rate limits
        # self.add_constraint('rw_speed', lower=-1, upper=1)

        # TODO: get angular speed from orbit
        # TODO: change variable of integration to nondimensionalized time
        osculating_orbit_angular_speed = self.declare_variable(
            'osculating_orbit_angular_speed',
            shape=(1, num_times),
            val=1,
            desc='Angular speed of oscullating orbit')

        v = np.concatenate((np.array([0, 0,
                                      1]), np.eye(3).flatten())).flatten()
        initial_angular_velocity_orientation = self.declare_variable(
            'initial_angular_velocity_orientation',
            shape=12,
            val=v,
            desc='Initial angular velocity in body frame')

        K1 = -0.5
        K2 = 0.9
        K3 = -(K1 + K2) / (1 + K1 * K2)

        angular_velocity_orientation = csdl.custom(
            external_torques_x,
            external_torques_y,
            external_torques_z,
            osculating_orbit_angular_speed,
            initial_angular_velocity_orientation,
            op=AttitudeRK4GravityComp(
                num_times=num_times,
                step_size=step_size,
                moment_inertia_ratios=np.array([K1, K2, K3]),
            ),
        )
        angular_velocity_orientation = self.register_output(
            'angular_velocity_orientation',
            angular_velocity_orientation,
        )

        # attitude outputs transpose of B_to_RTN
        RTN_to_B = self.register_output(
            'RTN_to_B',
            csdl.reshape(
                angular_velocity_orientation[3:, :],
                new_shape=(3, 3, num_times),
            ),
        )

        # ECI_to_B = csdl.einsum(ECI_to_RTN, RTN_to_B, 'ijm,klm->ilm')
        # B_to_ECI = csdl.einsum(ECI_to_B, 'ijk->jik')

        # earth_spin_rate = 2 * np.pi / 24 / 3600  # rad/s
        # t = self.declare_variable('t', val=np.arange(num_times) * step_size)
        # ECI_to_ESF = self.create_output('ECI_to_ESF',
        #                                 val=np.zeros((3, 3, num_times)))
        # ECI_to_ESF[0, 0, :] = csdl.cos(-earth_spin_rate * t)
        # ECI_to_ESF[0, 1, :] = csdl.sin(-earth_spin_rate * t)
        # ECI_to_ESF[1, 0, :] = csdl.cos(-earth_spin_rate * t)
        # ECI_to_ESF[1, 1, :] = -csdl.sin(-earth_spin_rate * t)
        # ECI_to_ESF[2, 2, :] = 1
        # B_to_ESF = csdl.einsum(B_to_ECI, ECI_to_ESF, 'ijm,klm->ilm')
        # ESF_to_B = csdl.einsum(B_to_ESF, 'ijk->jik')
        # sun_direction = csdl.reshape(ESF_to_B[:, 0, :], (3, num_times))

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


if __name__ == "__main__":
    from csdl_om import Simulator
    sim = Simulator(AttitudeGroup(num_times=1, step_size=0.1))
    sim.visualize_implementation()
