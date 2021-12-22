from csdl import Model
import csdl
import numpy as np


class RotationMatrices(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        earth_spin_rate = 2 * np.pi / 24 / 3600  # rad/s
        t = self.declare_variable('t', val=np.arange(num_times) * step_size)

        # Earth Centered Inertial to Earth Sun Frame
        v = np.zeros((3, 3, num_times))
        v[2, 2, :] = 1
        ECI_to_ESF = self.create_output('ECI_to_ESF', val=v)
        et = -earth_spin_rate * t
        ECI_to_ESF[0, 0, :] = csdl.cos(csdl.reshape(et, (1, 1, num_times)))
        ECI_to_ESF[0, 1, :] = csdl.sin(csdl.reshape(et, (1, 1, num_times)))
        ECI_to_ESF[1, 0, :] = csdl.cos(csdl.reshape(et, (1, 1, num_times)))
        ECI_to_ESF[1, 1, :] = -csdl.sin(csdl.reshape(et, (1, 1, num_times)))

        # Body to Earth Sun Frame
        B_to_ESF = csdl.einsum(B_to_ECI, ECI_to_ESF, subscripts='ijm,klm->ilm')

        # Earth Sun Frame to Body
        ESF_to_B = csdl.einsum(B_to_ESF, subscripts='ijk->jik')
        sun_direction = csdl.reshape(ESF_to_B[:, 0, :], (3, num_times))

        # orbit state given in ECI frame
        # TODO: connect orbit state
        orbit_state = self.declare_variable(
            'orbit_state',
            shape=(6, num_times),
        )
        radius = orbit_state[:3, :]
        velocity = orbit_state[:3, :]

        a0 = radius / csdl.expand(
            csdl.pnorm(radius, axis=0),
            (3, num_times),
            indices='i->ij',
        )
        tmp = csdl.cross(radius, velocity, axis=0)
        a2 = tmp / csdl.expand(
            csdl.pnorm(tmp, axis=0),
            (3, num_times),
            indices='i->ij',
        )
        a1 = csdl.cross(a2, a0, axis=0)

        ECI_to_RTN = self.create_output(
            'ECI_to_RTN',
            shape=(3, 3, num_times),
        )
        ECI_to_RTN[0, :, :] = csdl.expand(
            a0,
            (1, 3, num_times),
            indices='jk->ijk',
        )
        ECI_to_RTN[1, :, :] = csdl.expand(
            a1,
            (1, 3, num_times),
            indices='jk->ijk',
        )
        ECI_to_RTN[2, :, :] = csdl.expand(
            a2,
            (1, 3, num_times),
            indices='jk->ijk',
        )

        # TODO: attitude outputs transpose of B_to_RTN
        RTN_to_B = self.declare_variable(
            'RTN_to_B',
            shape=(3, 3, num_times),
        )

        ECI_to_B = csdl.einsum(ECI_to_RTN, RTN_to_B, 'ijm,klm->ilm')
        B_to_ECI = csdl.einsum(ECI_to_B, 'ijk->jik')

        earth_spin_rate = 2 * np.pi / 24 / 3600  # rad/s
        t = self.declare_variable('t', val=np.arange(num_times) * step_size)
        ECI_to_ESF = self.create_output('ECI_to_ESF',
                                        val=np.zeros((3, 3, num_times)))
        ECI_to_ESF[0, 0, :] = csdl.cos(-earth_spin_rate * t)
        ECI_to_ESF[0, 1, :] = csdl.sin(-earth_spin_rate * t)
        ECI_to_ESF[1, 0, :] = csdl.cos(-earth_spin_rate * t)
        ECI_to_ESF[1, 1, :] = -csdl.sin(-earth_spin_rate * t)
        ECI_to_ESF[2, 2, :] = 1
        B_to_ESF = csdl.einsum(B_to_ECI, ECI_to_ESF, 'ijm,klm->ilm')
        ESF_to_B = csdl.einsum(B_to_ESF, 'ijk->jik')
        sun_direction = csdl.reshape(ESF_to_B[:, 0, :], (3, num_times))
