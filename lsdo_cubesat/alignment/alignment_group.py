import numpy as np

from lsdo_cubesat.utils.decompose_vector_group import compute_norm_unit_vec
from lsdo_cubesat.alignment.sun_direction_comp import SunDirectionComp
from lsdo_cubesat.alignment.mask_vec_comp import MaskVecComp

from csdl import Model
import csdl


class Alignment(Model):
    def initialize(self):
        self.parameters.declare('swarm')

    def define(self):
        swarm = self.parameters['swarm']
        num_times = swarm['num_times']
        step_size = swarm['step_size']

        times = self.create_input(
            'times',
            val=np.linspace(0., step_size * (num_times - 1), num_times),
        )

        sun_unit_vec = csdl.custom(
            times,
            op=SunDirectionComp(
                num_times=num_times,
                launch_date=swarm['launch_date'],
            ),
        )

        # group = ConstantOrbitGroup(
        #     num_times=num_times,
        #     num_cp=num_cp,
        #     step_size=step_size,
        #     cubesat=swarm.children[0],
        # )
        # self.add('constant_orbit_group', group, promotes=['*'])
        position_unit_vec = self.declare_variable('position_unit_vec',
                                                  shape=(3, num_times))
        velocity_unit_vec = self.declare_variable('velocity_unit_vec',
                                                  shape=(3, num_times))
        normal_cross_vec = csdl.cross(
            velocity_unit_vec,
            position_unit_vec,
            axis=0,
        )
        observation_cross_vec = csdl.cross(
            position_unit_vec,
            sun_unit_vec,
            axis=0,
        )
        self.register_output('observation_cross_vec', observation_cross_vec)

        normal_cross_norm, normal_cross_unit_vec = compute_norm_unit_vec(
            normal_cross_vec,
            num_times=num_times,
        )

        observation_cross_norm, observation_cross_unit_vec = compute_norm_unit_vec(
            observation_cross_vec,
            num_times=num_times,
        )

        observation_dot = csdl.dot(
            observation_cross_unit_vec,
            normal_cross_unit_vec,
            axis=0,
        )
        self.register_output('observation_dot', observation_dot)

        mask_vec = csdl.custom(
            observation_dot,
            op=MaskVecComp(
                num_times=num_times,
                threshold=swarm['cross_threshold'],
                in_name='observation_dot',
                out_name='mask_vec',
            ),
        )

        # Separation

        separation_constraint_names = [
            # ('sunshade', 'optics'),
            ('optics', 'detector'),
        ]

        for name1, name2 in [
                # ('sunshade', 'optics'),
                # ('sunshade', 'detector'),
            ('optics', 'detector'),
        ]:
            position_name = 'position_{}_{}_km'.format(name1, name2)
            distance_name = 'distance_{}_{}_km'.format(name1, name2)
            unit_vec_name = 'unit_vec_{}_{}_km'.format(name1, name2)

            a = self.declare_variable(
                '{}_cubesat_group_position_km'.format(name1),
                shape=(3, num_times))
            b = self.declare_variable(
                '{}_cubesat_group_position_km'.format(name2),
                shape=(3, num_times))
            c = a - b
            self.register_output(position_name, c)

            d, pu = compute_norm_unit_vec(
                c,
                num_times=num_times,
            )
            self.register_output(distance_name, d)
            self.register_output(unit_vec_name, pu)

        # Transverse displacement

        transverse_constraint_names = [
            # ('sunshade', 'detector'),
            ('optics', 'detector'),
        ]

        for name1, name2 in transverse_constraint_names:
            position_name = 'position_{}_{}_km'.format(name1, name2)
            projected_position_name = 'projected_position_{}_{}_km'.format(
                name1, name2)
            normal_position_name = 'normal_position_{}_{}_km'.format(
                name1, name2)
            normal_distance_name = 'normal_distance_{}_{}_km'.format(
                name1, name2)
            normal_unit_vec_name = 'normal_unit_vec_{}_{}_km'.format(
                name1, name2)

            pos = self.declare_variable(position_name, shape=(3, num_times))

            c = sun_unit_vec * pos**2
            d = csdl.sum(c, axes=(0, ))
            e = csdl.expand(d, (3, num_times), 'i->ji')

            # self.register_output(projected_position_name, e)
            f = pos - e
            self.register_output(normal_position_name, f)

            g, h = compute_norm_unit_vec(
                f,
                num_times=num_times,
            )
            self.register_output(normal_distance_name, g)
            self.register_output(normal_unit_vec_name, h)

        for constraint_name in [
                'normal_distance_{}_{}'.format(name1, name2)
                for name1, name2 in transverse_constraint_names
        ] + [
                'distance_{}_{}'.format(name1, name2)
                for name1, name2 in separation_constraint_names
        ]:
            p = self.declare_variable(
                '{}_km'.format(constraint_name),
                shape=(num_times, ),
            )
            q = self.register_output('{}_mm'.format(constraint_name), 1.e6 * p)
            r = self.register_output('masked_{}_mm'.format(constraint_name),
                                     mask_vec * q)
            s = csdl.min(r, rho=100.)
            self.register_output('ks_masked_{}_mm'.format(constraint_name), s)

            t = self.register_output('masked_{}_mm_sq'.format(constraint_name),
                                     r**2)
            u = self.register_output(
                'masked_{}_mm_sq_sum'.format(constraint_name), csdl.sum(t))
