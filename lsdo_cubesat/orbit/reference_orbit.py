from lsdo_cubesat.orbit.keplerian_to_cartesian import KeplerianToCartesian
from ozone.api import ODEProblem
from csdl import Model
import csdl
import numpy as np

mu = 398600.44
Re = 6378.137
J2 = 1.08264e-3
J3 = -2.51e-6
J4 = -1.60e-6

A = -(3 / 2 * mu * J2 * Re**2)
B = -(5 / 2 * mu * J3 * Re**3)
C = (15 / 8 * mu * J4 * Re**4)


def accel_need_better_name(rmag, rz, pos):
    return (-mu / rmag**3 + A / rmag**5 *
            (1 - 5 * rz**2 / rmag**2) + B / rmag**7 *
            (3 * rz - 7 * rz**3 / rmag**2) + C / rmag**7 *
            (1 - 14 * rz**2 / rmag**2 + 21 * rz**4 / rmag**4)) * pos


def accel_z(rmag, rz):
    return (2 * A / rmag**5 + B / rmag**7 * (3 * rz) + C / rmag**7 *
            (4 - 28 / 3 * rz**2 / rmag**2)) * rz - 3 / 5 * B / rmag**5


class ReferenceOrbitDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        n = self.parameters['num_nodes']

        r = self.create_input('r', shape=(n, 6))
        rz = r[:, 2]
        pos = r[:, :3]

        tmp = csdl.pnorm(pos, axis=1)
        rmag = csdl.reshape(tmp, (n, 1))
        rz_n3 = csdl.expand(csdl.reshape(rz, (n, )), (n, 3), 'i->ij')
        rmag_n3 = csdl.expand(tmp, (n, 3), 'i->ij')

        a = accel_need_better_name(rmag_n3, rz_n3, pos)
        az = accel_z(rmag, rz)

        dr_dt = self.create_output('dr_dt', shape=(n, 6))
        dr_dt[:, :3] = r[:, 3:]
        dr_dt[:, 3] = a[:, 0]
        dr_dt[:, 4] = a[:, 1]
        dr_dt[:, 5] = a[:, 2] + az


class ReferenceOrbitTrajectory(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']

        eccentricity = 0.001
        # 450-550 km altitude
        periapsis = Re + 450.
        semimajor_axis = periapsis / (1 - eccentricity)

        self.add(
            KeplerianToCartesian(
                apoapsis=semimajor_axis * (1 + eccentricity),
                periapsis=semimajor_axis * (1 - eccentricity),
                longitude=66.279 * np.pi / 180.,
                # longitude=0. * np.pi / 180.,
                inclination=97.4 * np.pi / 180.,
                # inclination=0. * np.pi / 180.,
                argument_of_periapsis=0.,
                # argument_of_periapsis=45. * np.pi / 180.,
                # true_anomaly=337.987 * np.pi / 180.,
                true_anomaly=-90. * np.pi / 180.,
                r_name='initial_radius_km',
                v_name='initial_velocity_km_s',
            ),
            name='initial_orbit',
        )
        r = self.declare_variable('initial_radius_km', shape=(3, ))
        v = self.declare_variable('initial_velocity_km_s', shape=(3, ))

        r_0 = self.create_output('r_0', shape=(6, ))
        r_0[:3] = r
        r_0[3:] = v

        # self.create_input(
        #     'r_0',
        #     val=np.array([
        #         Re + 500.,
        #         0.,
        #         0.,
        #         0.,
        #         7.5,
        #         0.,
        #         # 1.76002146e+03,
        #         # 6.19179823e+03,
        #         # 6.31576531e+03,
        #         # 4.73422022e-05,
        #         # 1.26425269e-04,
        #         # 5.39731211e-05,
        #     ]),
        #     shape=(6, ))
        h = np.ones(num_times - 1) * step_size
        self.create_input('h', val=h)

        reference_orbit = ODEProblem('RK4', 'time-marching', num_times)
        reference_orbit.add_state('r',
                                  'dr_dt',
                                  shape=(6, ),
                                  initial_condition_name='r_0',
                                  output='reference_orbit_state_km')
        reference_orbit.add_times(step_vector='h')
        reference_orbit.set_ode_system(ReferenceOrbitDynamics)

        self.add(reference_orbit.create_solver_model())


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator

    num_times = 1500 + 1
    min = 90
    s = min * 60
    step_size = s / num_times
    print('step_size', step_size)

    rep = GraphRepresentation(
        ReferenceOrbitTrajectory(
            num_times=num_times,
            step_size=step_size,
        ), )
    sim = Simulator(rep, mode='rev')
    sim.run()
    # sim.compute_total_derivatives()
    # exit()
    # print(sim['reference_orbit_state_km'])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r = sim['reference_orbit_state_km']
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]
    ax.plot(x[0], y[0], z[0], 'o')
    ax.plot(x, y, z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
