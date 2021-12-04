import csdl
import numpy as np
from scipy.sparse import csc_matrix

from csdl import Model
import csdl

from lsdo_cubesat.operations.rk4_op import RK4


def skew(a, b, c):
    return np.array([
        [0, -c, b],
        [c, 0, -a],
        [-b, a, 0],
    ])


def skew_array(a):
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0],
    ])


class RWSpeed(RK4):
    def initialize(self):
        super().initialize()
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('rw_mmoi', types=np.ndarray)

        self.parameters['external_vars'] = [
            'body_torque',
            'body_rates',
        ]
        self.parameters['init_state_var'] = 'initial_rw_speed'
        self.parameters['state_var'] = 'rw_speed'

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        rw_mmoi = self.parameters['rw_mmoi']
        # external
        self.add_input(
            'body_torque',
            shape=(3, num_times),
        )
        self.add_input(
            'body_rates',
            shape=(3, num_times),
        )
        # initial
        self.add_input(
            'initial_rw_speed',
            val=0,
            shape=3,
        )
        # integrated
        self.add_output(
            'rw_speed',
            shape=(3, num_times),
        )
        self.dfdx = np.zeros((3, 6))
        self.dfdx[:, :3] = -np.eye(3)
        self.rw_mmoi = np.diag(rw_mmoi)
        self.rw_mmoi_inv = np.diag(1 / rw_mmoi)

    def f_dot(self, external, state):
        J = self.parameters['rw_mmoi']
        body_torque = external[:3]
        body_rates = external[3:6]
        return -np.matmul(self.rw_mmoi_inv,
                          (np.cross(body_rates, J * state) + body_torque))

    # ODE wrt state variables
    def df_dy(self, external, state):
        body_rates = external[3:6]
        return -np.matmul(self.rw_mmoi_inv,
                          np.matmul(skew_array(body_rates), self.rw_mmoi))

    # ODE wrt external inputs
    def df_dx(self, external, state):
        body_rates = external[3:6]
        Jw = np.matmul(self.rw_mmoi, body_rates)
        self.dfdx[:, 3:] = np.matmul(self.rw_mmoi_inv, skew_array(Jw))
        return self.dfdx


class Attitude(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        # TODO: values
        self.parameters.declare('max_rw_torque', default=0.007)
        self.parameters.declare('max_rw_power', default=0.007)
        self.parameters.declare('max_rw_speed', default=0.007)
        self.parameters.declare('sc_mmoi', types=np.ndarray)
        self.parameters.declare('rw_mmoi', types=np.ndarray)
        self.parameters.declare('gravity_gradient', types=bool)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        max_rw_torque = self.parameters['max_rw_torque']
        max_rw_speed = self.parameters['max_rw_speed']
        gravity_gradient = self.parameters['gravity_gradient']
        sc_mmoi = self.parameters['sc_mmoi']
        rw_mmoi = self.parameters['rw_mmoi']
        if sc_mmoi.shape != (3, ):
            raise ValueError(
                'sc_mmoi must have shape (3,); has shape {}'.format(
                    sc_mmoi.shape))

        if rw_mmoi.shape != (3, ):
            raise ValueError(
                'rw_mmoi must have shape (3,); has shape {}'.format(
                    rw_mmoi.shape))

        # --------------------------------------------------------------
        # Coordinate Reference frame changes

        # Earth Centered Inertial to Radial Tangential Normal
        # RTN frame is fixed in osculating orbit plane, so ECI_to_RTN
        # varies with time in all three axes
        ECI_to_RTN = self.declare_variable(
            'ECI_to_RTN',
            shape=(3, 3, num_times),
        )
        RTN_to_B = self.create_input(
            'RTN_to_B',
            shape=(3, 3, num_times),
        )
        # TODO: upper/lower?
        self.add_design_variable('RTN_to_B')

        # # Earth Centered Inertial to Body
        ECI_to_B = csdl.einsum(ECI_to_RTN, RTN_to_B, subscripts='ijm,klm->ilm')
        self.register_output('ECI_to_B', ECI_to_B)

        # # Rate of change of Reference frame transformation
        ECI_to_B_dot = self.create_output(
            'ECI_to_B_dot',
            val=0,
            shape=(3, 3, num_times),
        )
        ECI_to_B_dot[:, :, 1:] = (ECI_to_B[:, :, 1:] -
                                  ECI_to_B[:, :, :-1]) / step_size

        osculating_orbit_angular_speed = self.declare_variable(
            'osculating_orbit_angular_speed',
            shape=(1, num_times),
        )

        # Angular velocity of spacecraft in inertial frame
        # (skew symmetric cross operator)
        wcross = csdl.einsum(
            ECI_to_B_dot,
            # transpose
            csdl.einsum(ECI_to_B, subscripts='ijk->jik'),
            subscripts='ijl,jkl->ikl')
        body_rates = self.create_output('body_rates', shape=(3, num_times))
        body_rates[0, :] = csdl.reshape(wcross[2, 1, :], (1, num_times))
        body_rates[1, :] = csdl.reshape(wcross[0, 2, :], (1, num_times))
        body_rates[2, :] = csdl.reshape(wcross[1, 0, :], (1, num_times))

        # Angular acceleration of spacecraft in inertial frame
        body_accels = self.create_output(
            'body_accels',
            val=0,
            shape=(3, num_times),
        )
        body_accels[:,
                    1:] = (body_rates[:, 1:] - body_rates[:, :-1]) / step_size

        J = csc_matrix(np.diag(sc_mmoi), shape=(3, 3))
        Jw = body_rates * np.einsum('i,j->ij', sc_mmoi, np.ones(num_times))
        bt1 = self.create_output('bt1', shape=(3, num_times))
        bt1[0, :] = sc_mmoi[0] * body_accels[0, :]
        bt1[1, :] = sc_mmoi[1] * body_accels[1, :]
        bt1[2, :] = sc_mmoi[2] * body_accels[2, :]

        bt2 = csdl.cross(body_rates, Jw, axis=0)
        if gravity_gradient is True:
            # Terms associated with orientation of spacecraft used to
            # compute effect of gravity field on spacecraft angular momentum
            # over time
            bt3 = self.create_output('gravity_term', shape=(3, num_times))
            # bt3[0, :] = -3 * (sc_mmoi[1] - sc_mmoi[2]) * (csdl.reshape(
            #     RTN_to_B[0, 1, :] * RTN_to_B[0, 2, :],
            #     (1, num_times)) * osculating_orbit_angular_speed**2)
            # bt3[1, :] = -3 * (sc_mmoi[2] - sc_mmoi[0]) * (csdl.reshape(
            #     RTN_to_B[0, 2, :] * RTN_to_B[0, 0, :],
            #     (1, num_times)) * osculating_orbit_angular_speed**2)
            # bt3[2, :] = -3 * (sc_mmoi[0] - sc_mmoi[1]) * (csdl.reshape(
            #     RTN_to_B[0, 0, :] * RTN_to_B[0, 1, :],
            #     (1, num_times)) * osculating_orbit_angular_speed**2)

            # use transpose of RTN_to_B
            bt3[0, :] = -3 * (sc_mmoi[1] - sc_mmoi[2]) * (csdl.reshape(
                RTN_to_B[1, 0, :] * RTN_to_B[2, 0, :],
                (1, num_times)) * osculating_orbit_angular_speed**2)
            bt3[1, :] = -3 * (sc_mmoi[2] - sc_mmoi[0]) * (csdl.reshape(
                RTN_to_B[2, 0, :] * RTN_to_B[0, 0, :],
                (1, num_times)) * osculating_orbit_angular_speed**2)
            bt3[2, :] = -3 * (sc_mmoi[0] - sc_mmoi[1]) * (csdl.reshape(
                RTN_to_B[0, 0, :] * RTN_to_B[1, 0, :],
                (1, num_times)) * osculating_orbit_angular_speed**2)

            body_torque = bt1 + bt2 + bt3
        else:
            body_torque = bt1 + bt2
        self.register_output('body_torque', body_torque)

        initial_rw_speed = self.create_input(
            'initial_rw_speed',
            shape=(3, ),
            val=0,
        )
        rw_speed = csdl.custom(
            body_torque,
            body_rates,
            initial_rw_speed,
            op=RWSpeed(
                num_times=num_times,
                step_size=step_size,
                rw_mmoi=rw_mmoi,
            ),
        )
        self.register_output('rw_speed', rw_speed)
        rw_accel = self.create_output('rw_accel', shape=(3, num_times), val=0)
        rw_accel[:, 1:] = (rw_speed[:, 1:] - rw_speed[:, :-1]) / step_size

        rw_torque = self.create_output('rw_torque', shape=(3, num_times))
        rw_torque[0, :] = rw_mmoi[0] * rw_accel[0, :]
        rw_torque[1, :] = rw_mmoi[1] * rw_accel[1, :]
        rw_torque[2, :] = rw_mmoi[2] * rw_accel[2, :]

        # RW torque saturation
        self.add_constraint(
            'rw_torque',
            lower=-max_rw_torque,
            upper=max_rw_torque,
        )

        # RW rate saturation
        rw_speed_ks_min = csdl.min(rw_speed, axis=1, rho=20)
        rw_speed_ks_max = csdl.max(rw_speed, axis=1, rho=20)
        self.register_output('rw_speed_ks_min', rw_speed_ks_min)
        self.register_output('rw_speed_ks_max', rw_speed_ks_max)
        self.add_constraint('rw_speed_ks_min', lower=-max_rw_speed)
        self.add_constraint('rw_speed_ks_max', lower=max_rw_speed)


if __name__ == "__main__":
    from csdl_om import Simulator
    sim = Simulator(
        Attitude(
            num_times=3,
            step_size=0.1,
            sc_mmoi=np.array([18, 18, 6]) * 1e-3,
            rw_mmoi=28 * np.ones(3) * 1e-6,
            gravity_gradient=True,
        ))
    # sim.visualize_implementation()
    sim.check_partials(compact_print=True, method='fd')
