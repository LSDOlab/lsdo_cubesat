from csdl import Model
import csdl
import numpy as np

from csdl.utils.get_bspline_mtx import get_bspline_mtx
from lsdo_cubesat.attitude.attitude import Attitude
from lsdo_cubesat.propulsion.propulsion import Propulsion
from lsdo_cubesat.orbit.relative_orbit import RelativeOrbitTrajectory
from lsdo_cubesat.constants import RADII

Re = RADII['Earth']


class VehicleDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('initial_orbit_state', types=np.ndarray)
        self.parameters.declare('cubesat')

    def define(self):
        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        initial_orbit_state = self.parameters['initial_orbit_state']
        cubesat = self.parameters['cubesat']

        # time steps for all integrators
        h = np.ones(num_times - 1) * step_size
        self.create_input('h', val=h)

        # self.add(
        #     Propulsion(
        #         num_times=num_times,
        #         num_cp=num_cp,
        #         cubesat=cubesat,
        #     ),
        #     name='propulsion',
        # )
        # thrust = self.declare_variable('thrust', shape=(num_times, 3))
        # total_mass = self.declare_variable('total_mass', shape=(num_times, 3))
        # acceleration_due_to_thrust = thrust / total_mass
        # self.register_output('acceleration_due_to_thrust',
        #                     acceleration_due_to_thrust)

        acceleration_due_to_thrust_cp = self.create_input(
            'acceleration_due_to_thrust_cp',
            # val=(np.random.rand(3 * num_cp).reshape((num_cp, 3)) - 0.5) * 1e-4,
            # val=1e-11*np.random.rand(num_cp*3).reshape((num_cp, 3))*np.einsum('i,j->ij', np.exp(-np.arange(num_cp)), np.ones(3)),
            val=0,
            shape=(num_cp, 3),
        )
        self.add_design_variable('acceleration_due_to_thrust_cp', scaler=1e5)
        v = get_bspline_mtx(num_cp, num_times).toarray()
        bspline_mtx = self.declare_variable(
            'bspline_mtx',
            val=v,
            shape=v.shape,
        )
        acceleration_due_to_thrust = csdl.einsum(
            bspline_mtx,
            acceleration_due_to_thrust_cp,
            subscripts='kj,ji->ki',
        )
        self.register_output('acceleration_due_to_thrust',
                             acceleration_due_to_thrust)

        acceleration_due_to_thrust = self.create_input(
            'acceleration_due_to_thrust',
            val=0,
            shape=(num_times, 3),
        )
        self.add_design_variable('acceleration_due_to_thrust', scaler=1e5)
        self.add(
            RelativeOrbitTrajectory(
                num_times=num_times,
                step_size=step_size,
                initial_orbit_state=initial_orbit_state,
            ),
            name='orbit',
        )

        # orbit_state_km = self.declare_variable(
        #     'orbit_state_km',
        #     shape=(num_times, 6),
        # )
        # radius = orbit_state_km[:, :3]
        # velocity = orbit_state_km[:, 3:]

        # # altitude constraints
        # alt = csdl.pnorm(radius, axis=1) - Re
        # min_alt = csdl.min(alt)
        # max_alt = csdl.max(alt)
        # self.register_output('min_alt', min_alt)
        # self.register_output('max_alt', max_alt)
        # self.add_constraint('min_alt', lower=50.)
        # self.add_constraint('max_alt', upper=50.)

        # # NOTE: BOTH contain nans
        # self.register_output(
        #     'radius',
        #     radius,
        # )
        # self.register_output(
        #     'velocity',
        #     velocity,
        # )

        # a0 = radius / csdl.expand(
        #     csdl.pnorm(radius, axis=1),
        #     (num_times, 3),
        #     indices='i->ij',
        # )
        # osculating_orbit_angular_velocity = csdl.cross(radius,
        #                                                velocity,
        #                                                axis=1)
        # self.register_output(
        #     'osculating_orbit_angular_velocity',
        #     osculating_orbit_angular_velocity,
        # )
        # a2 = osculating_orbit_angular_velocity / csdl.expand(
        #     csdl.pnorm(osculating_orbit_angular_velocity, axis=1),
        #     (num_times, 3),
        #     indices='i->ij',
        # )
        # a1 = csdl.cross(a2, a0, axis=1)

        # RTN_from_ECI = self.create_output(
        #     'RTN_from_ECI',
        #     shape=(3, 3, num_times),
        # )
        # RTN_from_ECI[0, :, :] = csdl.expand(
        #     csdl.transpose(a0),
        #     (1, 3, num_times),
        #     indices='jk->ijk',
        # )
        # RTN_from_ECI[1, :, :] = csdl.expand(
        #     csdl.transpose(a1),
        #     (1, 3, num_times),
        #     indices='jk->ijk',
        # )
        # RTN_from_ECI[2, :, :] = csdl.expand(
        #     csdl.transpose(a2),
        #     (1, 3, num_times),
        #     indices='jk->ijk',
        # )

        # osculating_orbit_angular_speed = csdl.reshape(
        #     csdl.pnorm(osculating_orbit_angular_velocity, axis=1),
        #     (1, num_times))
        # self.register_output(
        #     'osculating_orbit_angular_speed',
        #     osculating_orbit_angular_speed,
        # )
        # self.add(
        #     Attitude(
        #         num_times=num_times,
        #         num_cp=num_cp,
        #         step_size=step_size,
        #         gravity_gradient=True,
        #     ),
        #     name='attitude',
        # )
