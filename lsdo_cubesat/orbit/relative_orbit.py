from ozone.api import ODEProblem
from csdl import Model
import csdl
import numpy as np
from lsdo_cubesat.orbit.reference_orbit import A, B, C, accel_need_better_name


class RelativeOrbitDynamics(Model):

    def initialize(self):
        self.parameters.declare('num_nodes', types=int)

    def define(self):
        n = self.parameters['num_nodes']

        # m, m/s
        rel = self.declare_variable('relative_orbit_state_m', shape=(n, 6))

        # dynamic parameters
        # km, km/s
        r = self.declare_variable('reference_orbit_state_km', shape=(n, 6))

        # thrust/mass; m/s**2
        acceleration_due_to_thrust = self.declare_variable(
            'acceleration_due_to_thrust', val=0, shape=(n, 3))

        rz = r[:, 2]

        # convert relative position and velocity to km, km/s
        u = rel[:, :3] / 1000

        # TODO: use true rmag?
        tmp = csdl.pnorm(r[:, :3] + u, axis=1)
        # tmp = csdl.pnorm(r[:, :3] + u, axis=1)
        rmag = csdl.reshape(tmp, (n, 1))
        rmag_n3 = csdl.expand(tmp, (n, 3), 'i->ij')

        uz = u[:, 2]
        rzuz = rz + uz
        rzuz_n3 = csdl.expand(csdl.reshape(rzuz, (n, )), (n, 3), 'i->ij')

        # Compute acceleration relative to reference orbit due to J2,
        # J3, J4 perturbations, not including (name?); convert to m/s
        a = accel_need_better_name(rmag_n3, rzuz_n3, u)

        # Compute (name?) acceleration in z direction relative to
        # reference orbit due to J2, J3, J4 perturbations; convert to
        # m/s;
        # More efficient version of:
        # from lsdo_cubesat.orbit.reference_orbit import accel_z
        # az = accel_z(rmag, rzuz) - accel_z(rmag, rz)
        # NOTE: Trajectory is very similar to using az = 0
        az = (2 * A / rmag**5 + 3 * B / rmag**7 * (2 * rz + uz) + C / rmag**7 *
              (4 - 28 / 3 / rmag**2 * (3 * rz**2 + 3 * rz * uz + uz**2))) * uz

        dr_dt = self.create_output('dr_dt', shape=(n, 6))
        dr_dt[:, :3] = rel[:, 3:]
        dr_dt[:, 3] = a[:, 0] + acceleration_due_to_thrust[:, 0]
        dr_dt[:, 4] = a[:, 1] + acceleration_due_to_thrust[:, 1]
        dr_dt[:, 5] = a[:, 2] + az + acceleration_due_to_thrust[:, 2]


class ODEProblemTest(ODEProblem):

    def setup(self):
        self.add_parameter('reference_orbit_state_km',
                           dynamic=True,
                           shape=(self.num_times, 6))
        self.add_parameter(
            'acceleration_due_to_thrust',
            dynamic=True,
            shape=(self.num_times, 3),
        )
        self.add_state(
            'relative_orbit_state_m',
            'dr_dt',
            shape=(6, ),
            initial_condition_name='rel_0',
            output='relative_orbit_state_m',
        )
        self.add_times(step_vector='h')
        self.set_ode_system(RelativeOrbitDynamics)


class RelativeOrbitTrajectory(Model):

    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('initial_orbit_state', types=np.ndarray)

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        initial_orbit_state = self.parameters['initial_orbit_state']

        reference_orbit_state_km = self.declare_variable(
            'reference_orbit_state_km', shape=(num_times, 6))
        acceleration_due_to_thrust = self.declare_variable(
            'acceleration_due_to_thrust',
            shape=(num_times, 3),
            val=0,
        )

        self.create_input('rel_0', val=initial_orbit_state, shape=(6, ))

        self.add(
            # ODEProblemTest('GaussLegendre4', 'collocation',
            ODEProblemTest('RK4', 'time-marching',
                           num_times).create_solver_model(),
            name='relative_orbit_integrator',
        )

        # TODO: add constraints to relative orbit states in telescope
        # config; probably don't need this anymore
        relative_orbit_state_m = self.declare_variable(
            'relative_orbit_state_m',
            shape=(num_times, 6),
        )

        orbit_state_km = reference_orbit_state_km + relative_orbit_state_m / 1000.
        self.register_output('orbit_state_km', orbit_state_km)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from csdl import GraphRepresentation
    from python_csdl_backend import Simulator
    from lsdo_cubesat.orbit.reference_orbit import ReferenceOrbitTrajectory

    num_times = 1500 + 1
    min = 360
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

    reference_orbit_state_km = sim['reference_orbit_state_km']
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    r = sim['reference_orbit_state_km']
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    rep = GraphRepresentation(
        RelativeOrbitTrajectory(
            num_times=num_times,
            step_size=step_size,
            initial_orbit_state=np.array([
                0,
                0,
                0,
                0.,
                0.,
                0,
            ]),
        ), )
    sim = Simulator(rep, mode='rev')
    print(r.shape)
    print(sim['reference_orbit_state_km'].shape)
    sim['reference_orbit_state_km'] = r
    sim.run()
    sim.compute_total_derivatives()
    exit()
    relative_orbit_state_m = sim['relative_orbit_state_m']
    orbit_state_km = sim['orbit_state_km']

    rx = relative_orbit_state_m[:, 0]
    ry = relative_orbit_state_m[:, 1]
    rz = relative_orbit_state_m[:, 2]

    axx = orbit_state_km[:, 0]
    ay = orbit_state_km[:, 1]
    az = orbit_state_km[:, 2]

    ax.plot(x, y, z)
    ax.plot(x + rx / 1000, y + ry / 1000, z + rz / 1000)
    ax.set_title('Reference and Absolute Orbit')
    ax.plot(axx, ay, az)
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot(rx, ry, rz)
    ax.set_title('Relative Orbit')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    plt.show()

    np.savetxt('relative_orbit', relative_orbit_state_m)
