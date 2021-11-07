import numpy as np

from csdl import Model, NonlinearBlockGS, LinearBlockGS
import csdl

from lsdo_cubesat.orbit.relative_orbit_rk4_comp import RelativeOrbitRK4Comp
from lsdo_cubesat.orbit.rot_mtx_t_i_comp import RotMtxTIComp
from lsdo_cubesat.utils.decompose_vector_group import compute_norm_unit_vec
from lsdo_cubesat.constants import RADII

radius_earth = RADII['Earth']


class Drag(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('cubesat')

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']

        # drag_unit_vec_t_3xn = self.declare_variable(
        #     'drag_unit_vec_t_3xn',
        #     val=np.outer(
        #         np.array([0., 0., 1.]),
        #         np.ones(num_times),
        #     ),
        # )
        drag_3xn = self.declare_variable(
            'drag_3xn',
            shape=(3, num_times),
            val=0,
        )
        thrust_3xn = self.declare_variable('thrust_3xn', shape=(3, num_times))
        radius = self.declare_variable('radius', shape=(1, num_times))
        mass = self.declare_variable('mass', shape=(1, num_times))
        force_3xn = thrust_3xn + drag_3xn
        initial_orbit_state = self.declare_variable(
            'initial_orbit_state',
            val=cubesat['initial_orbit_state'],
        )
        reference_orbit_state = self.declare_variable('reference_orbit_state',
                                                      shape=(6, num_times))

        # drag_scalar_3xn = self.declare_variable('drag_scalar_3xn',
        #                                         shape=(3, num_times))
        self.register_output('force_3xn', force_3xn)
        relative_orbit_state = csdl.custom(
            force_3xn,
            mass,
            radius,
            initial_orbit_state,
            op=RelativeOrbitRK4Comp(
                num_times=num_times,
                step_size=step_size,
            ),
        )
        relative_orbit_state = self.register_output('relative_orbit_state',
                                                    relative_orbit_state)
        orbit_state = relative_orbit_state + reference_orbit_state
        orbit_state_km = orbit_state * 1.e-3
        self.register_output('orbit_state_km', orbit_state_km)

        # TODO: Why was Aobo taking the norm of the entire state and not
        # only the position relative to the reference orbit?
        relative_orbit_state_sq = relative_orbit_state[:3, :]**2

        relative_orbit_state_sq_sum = csdl.sum(relative_orbit_state_sq)
        self.register_output('relative_orbit_state_sq_sum',
                             relative_orbit_state_sq_sum)

        # TODO: restore
        # comp = RotMtxTIComp(num_times=num_times)
        # coupled_group.add('rot_mtx_t_i_3x3xn_comp', comp, promotes=['*'])
        rot_mtx_t_i_3x3xn = self.declare_variable('rot_mtx_t_i_3x3xn',
                                                  shape=(3, 3, num_times))

        rot_mtx_i_t_3x3xn = csdl.reorder_axes(
            rot_mtx_t_i_3x3xn,
            'ijn->jin',
        )

        # drag_unit_vec_3xn = csdl.einsum(
        #     rot_mtx_i_t_3x3xn,
        #     drag_unit_vec_t_3xn,
        #     subscripts='ijk,jk->ik',
        # )

        # r = drag_3xn - drag_unit_vec_3xn * drag_scalar_3xn
        # self.register_output('r', r)

        # coupled_group.nonlinear_solver = NonlinearBlockGS(iprint=0,
        #                                                   maxiter=40,
        #                                                   atol=1e-14,
        #                                                   rtol=1e-12)
        # coupled_group.linear_solver = LinearBlockGS(iprint=0,
        #                                             maxiter=40,
        #                                             atol=1e-14,
        #                                             rtol=1e-12)

        # self.add('coupled_group', coupled_group, promotes=['*'])


class OrbitGroup(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('cubesat')

    def define(self):
        num_times = self.parameters['num_times']
        step_size = self.parameters['step_size']
        cubesat = self.parameters['cubesat']

        dry_mass = self.create_input('dry_mass',
                                     val=cubesat['dry_mass'],
                                     shape=num_times)
        initial_orbit_state = self.create_input(
            'initial_orbit_state',
            val=cubesat['initial_orbit_state'],
        )

        propellant_mass = self.declare_variable(
            'propellant_mass',
            shape=num_times,
        )

        mass = csdl.expand(dry_mass + propellant_mass, (1, num_times),
                           indices='i->ji')
        self.register_output('mass', mass)
        self.add(
            Drag(
                num_times=num_times,
                step_size=step_size,
                cubesat=cubesat,
            ), )

        # will need nested implicit operations because attitude requires
        # mean motion, and drag requires attitude
        # orbit_op = self.create_implicit_operation(
        #     Drag(
        #         num_times=num_times,
        #         step_size=step_size,
        #         cubesat=cubesat,
        #     ))
        # orbit_op.declare_state('drag_3xn', residual='r')

        radius = self.declare_variable('radius', shape=(1, num_times))
        thrust_3xn = self.declare_variable(
            'thrust_3xn',
            shape=(3, num_times),
        )
        reference_orbit_state = self.declare_variable(
            'reference_orbit_state',
            shape=(6, num_times),
        )

        # drag_scalar_3xn = self.declare_variable(
        #     'drag_scalar_3xn',
        #     shape=(3, num_times),
        #     val=0,
        # )

        # NOTE: edges are saved
        # drag_3xn, orbit_state_km, relative_orbit_state =
        # drag_3xn = orbit_op(
        #     mass,
        #     initial_orbit_state,
        #     thrust_3xn,
        #     reference_orbit_state,
        #     drag_scalar_3xn,
        #     radius,
        #     expose=['orbit_state_km', 'relative_orbit_state'],
        # )

        orbit_state_km = self.declare_variable('orbit_state_km',
                                               shape=(6, num_times))
        position_km = orbit_state_km[:3, :]
        velocity_km_s = orbit_state_km[3:, :]

        position = position_km * 1.e3
        velocity = velocity_km_s * 1.e3

        radius_km, position_unit_vec = compute_norm_unit_vec(
            position_km, num_times=num_times)
        speed_km_s, velocity_unit_vec = compute_norm_unit_vec(
            velocity_km_s, num_times=num_times)

        altitude_km = radius_km - radius_earth
        ks_altitude_km = csdl.min(
            altitude_km,
            rho=100.,
        )

        self.register_output('ks_altitude_km', ks_altitude_km)
        self.add_constraint('ks_altitude_km', lower=450.)
