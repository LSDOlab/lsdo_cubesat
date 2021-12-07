import numpy as np

from lsdo_cubesat.utils.compute_norm_unit_vec import compute_norm_unit_vec
from lsdo_cubesat.operations.mask import Mask

from lsdo_cubesat.constants import deg2arcsec
from csdl import Model
import csdl


class TelescopeConfiguration(Model):
    def initialize(self):
        self.parameters.declare('swarm')
        self.parameters.declare('telescope_length_m', default=40., types=float)
        self.parameters.declare('telescope_length_tol_mm',
                                default=15.,
                                types=float)
        self.parameters.declare('telescope_view_plane_tol_mm',
                                default=8.,
                                types=float)
        # constrain each s/c to satisfy pointing accuracy
        self.parameters.declare('telescope_view_halfangle_tol_arcsec',
                                default=90.,
                                types=float)
        self.parameters.declare('relative_speed_tol_um_s',
                                default=100.,
                                types=float)

    def define(self):
        swarm = self.parameters['swarm']
        telescope_length_m = self.parameters['telescope_length_m']
        telescope_length_tol_mm = self.parameters['telescope_length_tol_mm']
        telescope_view_plane_tol_mm = self.parameters[
            'telescope_view_plane_tol_mm']
        num_times = swarm['num_times']
        telescope_view_halfangle_tol_arcsec = self.parameters[
            'telescope_view_halfangle_tol_arcsec']
        relative_speed_tol_um_s = self.parameters['relative_speed_tol_um_s']

        if telescope_length_m <= 0:
            raise ValueError('Telescope length must be a positive value')
        if telescope_length_tol_mm <= 0:
            raise ValueError(
                'Telescope length tolerance must be a positive value')

        optics_orbit_state_km = self.declare_variable('optics_orbit_state_km',
                                                      shape=(6, num_times))
        detector_orbit_state_km = self.declare_variable(
            'detector_orbit_state_km', shape=(6, num_times))

        _, optics_velocity_unit_vec = compute_norm_unit_vec(
            optics_orbit_state_km[:3, :], num_times=num_times)
        _, detector_velocity_unit_vec = compute_norm_unit_vec(
            detector_orbit_state_km[:3, :], num_times=num_times)
        sun_direction = self.declare_variable('sun_direction',
                                              shape=(3, num_times))
        optics_B_from_ECI = self.declare_variable('optics_B_from_ECI',
                                                  shape=(3, 3, num_times))
        detector_B_from_ECI = self.declare_variable('detector_B_from_ECI',
                                                    shape=(3, 3, num_times))
        optics_sun_direction_body = csdl.einsum(optics_B_from_ECI,
                                                sun_direction,
                                                subscripts='ijk,jk->ik')
        detector_sun_direction_body = csdl.einsum(detector_B_from_ECI,
                                                  sun_direction,
                                                  subscripts='ijk,jk->ik')
        # define observation phase as time when both s/c are flying
        # towards the sun
        optics_observation_dot = csdl.dot(
            optics_velocity_unit_vec,
            sun_direction,
            axis=0,
        )
        detector_observation_dot = csdl.dot(
            detector_velocity_unit_vec,
            sun_direction,
            axis=0,
        )
        self.register_output(
            'optics_observation_dot',
            optics_observation_dot,
        )
        self.register_output(
            'detector_observation_dot',
            detector_observation_dot,
        )

        # apply additional filter to ensure sun_LOS == 1
        # optics_sun_LOS = self.declare_variable('optics_sun_LOS',
        #                                        shape=(1, num_times))
        # detector_sun_LOS = self.declare_variable('detector_sun_LOS',
        #                                          shape=(1, num_times))

        optics_observation_phase_indicator = csdl.custom(
            optics_observation_dot,
            op=Mask(
                num_times=num_times,
                threshold=swarm['cross_threshold'],
                in_name='optics_observation_dot',
                out_name='optics_observation_phase_indicator',
            ),
        )
        detector_observation_phase_indicator = csdl.custom(
            detector_observation_dot,
            op=Mask(
                num_times=num_times,
                threshold=swarm['cross_threshold'],
                in_name='detector_observation_dot',
                out_name='detector_observation_phase_indicator',
            ),
        )
        self.register_output(
            'optics_observation_phase_indicator',
            optics_observation_phase_indicator,
        )
        self.register_output(
            'detector_observation_phase_indicator',
            detector_observation_phase_indicator,
        )
        observation_phase_indicator = optics_observation_phase_indicator * detector_observation_phase_indicator
        self.register_output('observation_phase_indicator',
                             observation_phase_indicator)

        # limit relative speed during observation
        optics_relative_orbit_state_m = self.declare_variable(
            'optics_relative_orbit_state_m', shape=(6, num_times))
        detector_relative_orbit_state_m = self.declare_variable(
            'detector_relative_orbit_state_m', shape=(6, num_times))
        relative_velocity_m_s = optics_relative_orbit_state_m[
            3:, :] - detector_relative_orbit_state_m[3:, :]
        relative_speed_m_s = csdl.pnorm(relative_velocity_m_s, axis=0)
        relative_speed_um_s = relative_speed_m_s * 1e6
        max_relative_speed_um_s = csdl.max(observation_phase_indicator *
                                           relative_speed_um_s)
        self.register_output('max_relative_speed_um_s',
                             max_relative_speed_um_s)
        self.add_constraint('max_relative_speed_um_s',
                            upper=relative_speed_tol_um_s)

        # Separation

        # NOTE: compute separation in terms of positions relative to
        # reference orbit to satisfy constraints on order of mm when
        # radius is on order of thousands of km
        optics_cubesat_relative_position = optics_relative_orbit_state_m[:3, :]
        detector_cubesat_relative_position = detector_relative_orbit_state_m[:
                                                                             3, :]
        self.register_output(
            'optics_cubesat_relative_position',
            optics_cubesat_relative_position,
        )
        self.register_output(
            'detector_cubesat_relative_position',
            detector_cubesat_relative_position,
        )

        telescope_vector = (optics_cubesat_relative_position -
                            detector_cubesat_relative_position)
        separation_m, telescope_direction = compute_norm_unit_vec(
            telescope_vector,
            num_times=num_times,
        )
        self.register_output('separation_m', separation_m)
        self.register_output('telescope_direction', telescope_direction)

        min_separation_error = csdl.min(observation_phase_indicator *
                                        (separation_m - telescope_length_m))
        max_separation_error = csdl.max(observation_phase_indicator *
                                        (separation_m - telescope_length_m))
        self.register_output('min_separation_error', min_separation_error)
        self.register_output('max_separation_error', max_separation_error)
        self.add_constraint(
            'min_separation_error',
            lower=-telescope_length_tol_mm,
        )
        self.add_constraint(
            'max_separation_error',
            upper=telescope_length_tol_mm,
        )

        # Transverse displacement

        # NOTE: compute view plane error in terms of positions repative
        # to reference orbit to satisfy constraints on order of mm when
        # radius is on order of thousands of km
        view_plane_error = csdl.pnorm(
            telescope_vector - csdl.expand(csdl.einsum(
                telescope_vector,
                sun_direction,
                subscripts='ij,ij->j',
            ), (3, num_times),
                                           indices='i->ji'),
            axis=0,
        )
        min_view_plane_error = csdl.min(observation_phase_indicator *
                                        view_plane_error)
        max_view_plane_error = csdl.max(observation_phase_indicator *
                                        view_plane_error)

        self.register_output('min_view_plane_error', min_view_plane_error)
        self.register_output('max_view_plane_error', max_view_plane_error)
        self.add_constraint('min_view_plane_error',
                            lower=-telescope_view_plane_tol_mm / 1000.)
        self.add_constraint('max_view_plane_error',
                            upper=-telescope_view_plane_tol_mm / 1000.)

        # Orientation during observation

        optics_cos_view_angle_error = -csdl.reshape(
            optics_sun_direction_body[1, :], (num_times, ))
        detector_cos_view_angle_error = -csdl.reshape(
            detector_sun_direction_body[0, :], (num_times, ))
        a = observation_phase_indicator * optics_cos_view_angle_error
        b = observation_phase_indicator * detector_cos_view_angle_error

        # KLUDGE: min/max not working as expected for arrays;
        # retrun variable with shape == ()
        max_optics_view_angle_error = deg2arcsec * csdl.arccos(
            csdl.max(csdl.reshape(1 - a, (1, num_times)), axis=1))
        max_detector_view_angle_error = deg2arcsec * csdl.arccos(
            csdl.max(csdl.reshape(1 - b, (1, num_times)), axis=1))

        self.register_output(
            'max_optics_view_angle_error',
            max_optics_view_angle_error,
        )
        self.register_output(
            'max_detector_view_angle_error',
            max_detector_view_angle_error,
        )
        self.add_constraint('max_optics_view_angle_error',
                            upper=telescope_view_halfangle_tol_arcsec)
        self.add_constraint('max_detector_view_angle_error',
                            upper=telescope_view_halfangle_tol_arcsec)

        # # =============================================================
        # # =============================================================
        # # transverse_constraint_names = [
        # #     # ('sunshade', 'detector'),
        # #     ('optics', 'detector'),
        # # ]

        # # for name1, name2 in transverse_constraint_names:
        # #     position_name = 'position_{}_{}_km'.format(name1, name2)
        # #     projected_position_name = 'projected_position_{}_{}_km'.format(
        # #         name1, name2)
        # #     normal_position_name = 'normal_position_{}_{}_km'.format(
        # #         name1, name2)
        # #     normal_distance_name = 'normal_distance_{}_{}_km'.format(
        # #         name1, name2)
        # #     normal_unit_vec_name = 'normal_unit_vec_{}_{}_km'.format(
        # #         name1, name2)

        # #     pos = self.declare_variable(position_name, shape=(3, num_times))

        # #     c = sun_direction * pos**2
        # #     d = csdl.sum(c, axes=(0, ))
        # #     e = csdl.expand(d, (3, num_times), 'i->ji')

        # #     # self.register_output(projected_position_name, e)
        # #     f = pos - e
        # #     self.register_output(normal_position_name, f)

        # #     g, h = compute_norm_unit_vec(
        # #         f,
        # #         num_times=num_times,
        # #     )
        # #     self.register_output(normal_distance_name, g)
        # #     self.register_output(normal_unit_vec_name, h)

        # # for constraint_name in [
        # #         'normal_distance_{}_{}'.format(name1, name2)
        # #         for name1, name2 in transverse_constraint_names
        # # ] + [
        # #         'distance_{}_{}'.format(name1, name2)
        # #         for name1, name2 in separation_constraint_names
        # # ]:
        # #     p = self.declare_variable(
        # #         '{}_km'.format(constraint_name),
        # #         shape=(num_times, ),
        # #     )
        # #     q = self.register_output('{}_mm'.format(constraint_name), 1.e6 * p)
        # #     r = self.register_output('masked_{}_mm'.format(constraint_name),
        # #                              mask_vec * q)
        # #     s = csdl.min(r, rho=100.)
        # #     self.register_output('ks_masked_{}_mm'.format(constraint_name), s)

        # #     t = self.register_output('masked_{}_mm_sq'.format(constraint_name),
        # #                              r**2)
        # #     u = self.register_output(
        # #         'masked_{}_mm_sq_sum'.format(constraint_name), csdl.sum(t))


if __name__ == '__main__':
    from csdl_om import Simulator
    from lsdo_cubesat.parameters.cubesat import CubesatParams
    from lsdo_cubesat.parameters.swarm import SwarmParams

    # initial state relative to reference orbit -- same for all s/c
    initial_orbit_state = np.array([1e-3] * 3 + [1e-3] * 3)

    cubesats = dict()

    cubesats['optics'] = CubesatParams(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6),
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    )

    cubesats['detector'] = CubesatParams(
        name='detector',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6),
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.002,
        apogee_altitude=499.98,
    )
    swarm = SwarmParams(
        num_times=4,
        num_cp=3,
        step_size=95 * 60 / (4 - 1),
        cross_threshold=0.857,
    )
    for v in cubesats.values():
        swarm.add(v)
    sim = Simulator(TelescopeConfiguration(swarm=swarm, ))
    sim.visualize_implementation()
