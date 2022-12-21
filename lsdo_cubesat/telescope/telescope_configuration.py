import numpy as np

from lsdo_cubesat.csdl_future.mask import MaskGE, MaskLT, MaskGT

from lsdo_cubesat.constants import deg2arcsec, s
from csdl import Model, GraphRepresentation
import csdl

from sys import float_info

# TODO: Use the "NOTE" comments in this file to design error messages
# where domain of finite derivatives is not the same as domain of valid
# inputs for individual functions; raise error if these functions are
# used and no runtime assertions provided; warn if runtime assertions
# are either too restrictive or guaranteed to pass; runtime assertions
# are required (1) for the compiler to verify that the user has
# communicated how the standard library functions are to be used, and
# (2) so that other users can be reminded of valid domains for inputs


class TelescopeConfiguration(Model):

    def initialize(self):
        self.parameters.declare('swarm')
        self.parameters.declare('telescope_length_m', default=40., types=float)
        self.parameters.declare('telescope_length_tol_mm',
                                default=15.,
                                types=float)
        self.parameters.declare('telescope_view_plane_tol_mm',
                                default=18.,
                                types=float)
        # constrain telescope and each s/c to satisfy pointing accuracy
        self.parameters.declare('telescope_view_halfangle_tol_arcsec',
                                default=90.,
                                types=float)
        # TODO: for a later paper, find appropriate speed constraint
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

        # optics_relative_orbit_state_m = self.declare_variable(
        # 'optics_relative_orbit_state_m', shape=(num_times, 6))
        # detector_relative_orbit_state_m = self.declare_variable(
        # 'detector_relative_orbit_state_m', shape=(num_times, 6))
        optics_orbit_state_km = self.declare_variable('optics_orbit_state_km',
                                                      shape=(num_times, 6))
        detector_orbit_state_km = self.declare_variable(
            'detector_orbit_state_km', shape=(num_times, 6))

        sun_direction = self.declare_variable('sun_direction',
                                              shape=(num_times, 3))

        # define observation phase as time when both s/c are flying
        # towards the sun; magnitude is not important except for setting
        # minimum threshold for indicating observation phase
        optics_observation_dot = csdl.reshape(optics_orbit_state_km[:, 3],
                                              (num_times, ))
        detector_observation_dot = csdl.reshape(detector_orbit_state_km[:, 3],
                                                (num_times, ))
        self.register_output(
            'optics_observation_dot',
            optics_observation_dot,
        )
        self.register_output(
            'detector_observation_dot',
            detector_observation_dot,
        )

        # define observation phase as time when both s/c are flying
        # towards the sun and have line of sight; use mask operation to
        # limit observation to time when telescope is sufficiently
        # aligned
        optics_sun_LOS = self.declare_variable('optics_sun_LOS',
                                               shape=(num_times, 1))
        detector_sun_LOS = self.declare_variable('detector_sun_LOS',
                                                 shape=(num_times, 1))
        los = csdl.reshape(optics_sun_LOS * detector_sun_LOS, (num_times, ))
        self.register_output('los', los)
        telescope_sun_LOS_indicator = csdl.custom(
            los,
            op=MaskGE(
                num_times=num_times,
                threshold=1,
                in_name='los',
                out_name='telescope_sun_LOS_indicator',
            ),
        )
        optics_observation_phase_indicator = csdl.custom(
            optics_observation_dot,
            op=MaskGE(
                num_times=num_times,
                threshold=swarm['cross_threshold'],
                in_name='optics_observation_dot',
                out_name='optics_observation_phase_indicator',
            ),
        )
        detector_observation_phase_indicator = csdl.custom(
            detector_observation_dot,
            op=MaskGE(
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
        observation_phase_indicator = optics_observation_phase_indicator * detector_observation_phase_indicator * telescope_sun_LOS_indicator
        self.register_output('observation_phase_indicator',
                             observation_phase_indicator)

        # limit relative speed during observation
        optics_relative_orbit_state_m = self.declare_variable(
            'optics_relative_orbit_state_m', shape=(num_times, 6))
        detector_relative_orbit_state_m = self.declare_variable(
            'detector_relative_orbit_state_m', shape=(num_times, 6))
        relative_velocity_m_s = optics_relative_orbit_state_m[:,
                                                              3:] - detector_relative_orbit_state_m[:,
                                                                                                    3:]
        relative_speed_m_s = csdl.pnorm(relative_velocity_m_s, axis=0)
        relative_speed_um_s = relative_speed_m_s * 1e6

        # TODO: for a later paper, include constraints on relative speed
        # max_relative_speed_um_s = csdl.max(observation_phase_indicator *
        #                                    relative_speed_um_s,rho=10.)
        # self.register_output('max_relative_speed_um_s',
        #                      max_relative_speed_um_s)
        # self.add_constraint('max_relative_speed_um_s',
        #                     upper=relative_speed_tol_um_s)

        # Separation

        # NOTE: compute separation in terms of positions relative to
        # reference orbit to satisfy constraints on order of mm when
        # radius is on order of thousands of km
        telescope_vector = (optics_relative_orbit_state_m[:, :3] -
                            detector_relative_orbit_state_m[:, :3])
        self.register_output('telescope_vector', telescope_vector)

        # NOTE: if spacecraft start from different initial positions,
        # this is highly unlikely to be zero, i.e. nondifferentiable
        separation_m = csdl.pnorm(telescope_vector, axis=1)
        self.register_output('separation_m', separation_m)
        separation_error = (separation_m - telescope_length_m)**2
        self.register_output('separation_error', separation_error)
        separation_error_during_observation = observation_phase_indicator * separation_error
        self.register_output('separation_error_during_observation',
                             separation_error_during_observation)
        max_separation_error_during_observation = csdl.max(
            separation_error_during_observation)
        self.register_output(
            'max_separation_error_during_observation',
            max_separation_error_during_observation,
        )
        self.add_constraint('max_separation_error_during_observation',
                            upper=(telescope_length_tol_mm / 1000.)**2,
                            scaler=s)
        # max_separation = csdl.max(observation_phase_indicator * separation_m)
        # self.register_output('max_separation', max_separation)
        # self.add_constraint(
        #     'max_separation',
        #     upper=(40. + telescope_length_tol_mm / 1000.)**2,
        # )
        # # NOTE: DO NOT CHANGE rho!!
        # min_separation_error = csdl.min(separation_error_during_observation,
        #                                 # rho=10. / 1e0,
        #                                 )
        # # NOTE: DO NOT CHANGE rho!!
        # max_separation_error = csdl.max(
        #     separation_error_during_observation,
        #     # separation_error_during_observation * 1e3,
        #     # rho=10. / 1e-1,
        # )
        # self.register_output('min_separation_error', min_separation_error)
        # self.register_output('max_separation_error', max_separation_error)
        # self.add_constraint(
        #     'min_separation_error',
        #     lower=-telescope_length_tol_mm / 1000.,
        # )
        # self.add_constraint(
        #     'max_separation_error',
        #     upper=telescope_length_tol_mm / 1000.,
        # )

        # Constrain view plane error to meet requirements
        telescope_vector_component_in_sun_direction = self.create_output(
            'telescope_vector_component_in_sun_direction',
            shape=(num_times, 3),
            val=0,
        )
        telescope_vector_component_in_sun_direction[:,
                                                    0] = telescope_vector[:,
                                                                          0] * sun_direction[:,
                                                                                             0]
        # telescope_vector_component_in_sun_direction[:,
        #                                             1] = telescope_vector[:,
        #                                                                   1] * sun_direction[:,
        #                                                                                      1]
        # telescope_vector_component_in_sun_direction[:,
        #                                             2] = telescope_vector[:,
        #                                                                   2] * sun_direction[:,
        #                                                                                      2]

        telescope_direction_in_view_plane = telescope_vector - telescope_vector_component_in_sun_direction
        self.register_output('telescope_direction_in_view_plane',
                             telescope_direction_in_view_plane)
        view_plane_error = csdl.sum(
            (telescope_direction_in_view_plane)**2,
            axes=(1, ),
        )
        self.register_output('view_plane_error', view_plane_error)
        view_plane_error_during_observation = observation_phase_indicator * view_plane_error
        self.register_output('view_plane_error_during_observation',
                             view_plane_error_during_observation)

        max_view_plane_error = csdl.max(
            view_plane_error_during_observation,
            rho=50.,
        )

        self.register_output('max_view_plane_error', max_view_plane_error)
        self.add_constraint(
            'max_view_plane_error',
            upper=(telescope_view_plane_tol_mm / 1000.)**2,
            scaler=100,
        )

        # # Orientation of telescope during observation (INACCURATE)
        # cos_view_angle = csdl.dot(
        #     sun_direction,
        #     telescope_vector,
        #     axis=1,
        # ) / csdl.pnorm(telescope_vector, axis=1)
        # self.register_output('telescope_cos_view_angle', cos_view_angle)
        # telescope_view_angle_unmasked = csdl.arccos(cos_view_angle)
        # telescope_view_angle = observation_phase_indicator * telescope_view_angle_unmasked

        # Orientation of telescope during observation (CONTINGENCY)
        cos_view_angle = csdl.dot(
            sun_direction,
            telescope_vector,
            axis=1,
        ) / telescope_length_m
        self.register_output('cos_view_angle', cos_view_angle)

        # NOTE: need to be able to take arccos safely, so only use
        # values where -1 < cos_view_angle < 1; when cos_view_angle >=
        # 1, separation constraint is violated; if separation constraint
        # is satsified, then cos_view_angle <= 1; finite derivatives
        # where -1 < cos_view_angle < 1
        cos_view_angle_lt1 = csdl.custom(cos_view_angle,
                                         op=MaskLT(
                                             num_times=num_times,
                                             threshold=1.,
                                             in_name='cos_view_angle',
                                             out_name='cos_view_angle_lt1',
                                         ))
        cos_view_angle_gt_neg1 = csdl.custom(
            cos_view_angle,
            op=MaskGT(
                num_times=num_times,
                threshold=-1.,
                in_name='cos_view_angle',
                out_name='cos_view_angle_gt_neg1',
            ))

        self.register_output('cos_view_angle_lt1', cos_view_angle_lt1)
        self.register_output('cos_view_angle_gt_neg1', cos_view_angle_gt_neg1)
        mag_cos_view_angle_lt1 = cos_view_angle_lt1 * cos_view_angle_gt_neg1
        telescope_cos_view_angle = mag_cos_view_angle_lt1 * cos_view_angle
        self.register_output(
            'telescope_cos_view_angle',
            telescope_cos_view_angle,
        )

        # NOTE: with good initial conditions, angle will be nonnegative
        # after taking arccos AND zeroing values outside observation
        # phase
        mag_cos_view_angle_ge1 = (1 - mag_cos_view_angle_lt1)
        cos_view_angle_le_neg1 = (1 - cos_view_angle_gt_neg1)

        telescope_view_angle_unmasked = csdl.arccos(mag_cos_view_angle_lt1 *
                                                    telescope_cos_view_angle)
        telescope_view_angle = observation_phase_indicator * (
            telescope_view_angle_unmasked -
            (np.pi / 2 - float_info.min) * mag_cos_view_angle_ge1 +
            (np.pi - float_info.min) * cos_view_angle_le_neg1)

        # END OF INACCURATE VS CONTINGENCY MODELING

        self.register_output(
            'telescope_view_angle_unmasked',
            telescope_view_angle_unmasked,
        )
        self.register_output(
            'telescope_view_angle',
            telescope_view_angle,
        )
        max_telescope_view_angle = csdl.max(telescope_view_angle)
        self.register_output(
            'max_telescope_view_angle',
            max_telescope_view_angle,
        )
        self.add_constraint(
            'max_telescope_view_angle',
            upper=telescope_view_halfangle_tol_arcsec / deg2arcsec * np.pi /
            180.,
            scaler=telescope_view_halfangle_tol_arcsec /4.85e-6,
        )

        attitude = True
        if attitude is True:
            # optics cubesat orients so that -y-axis is aligned with sun
            # detector cubesat orients so that -x-axis is aligned with sun
            # -optics_B_from_ECI[:,0,:][1, :] -> -optics_B_from_ECI[1,0,:]
            # -(sr * sp * cy - cr * sy) == 1
            # -detector_B_from_ECI[:,0,:][0, :] -> -detector_B_from_ECI[0,0,:]
            # -(cp * cy) == 1

            # Orientation of each s/c during observation
            optics_B_from_ECI = self.declare_variable('optics_B_from_ECI',
                                                      shape=(3, 3, num_times))
            detector_B_from_ECI = self.declare_variable('detector_B_from_ECI',
                                                        shape=(3, 3,
                                                               num_times))

            optics_sun_direction_body = csdl.einsum(optics_B_from_ECI,
                                                    sun_direction,
                                                    subscripts='ijk,kj->ik')
            detector_sun_direction_body = csdl.einsum(detector_B_from_ECI,
                                                      sun_direction,
                                                      subscripts='ijk,kj->ik')

            # # optics and detector oriented differently on each s/c
            # optics_cos_view_angle = csdl.reshape(
            #     optics_sun_direction_body[1, :], (num_times, ))
            # detector_cos_view_angle = csdl.reshape(
            #     detector_sun_direction_body[0, :], (num_times, ))
            # self.register_output('optics_cos_view_angle',
            #                      optics_cos_view_angle)
            # self.register_output('detector_cos_view_angle',
            #                      detector_cos_view_angle)
            # optics_cos_view_angle_during_observation = observation_phase_indicator * (
            #     optics_cos_view_angle - 1) + 1
            # detector_cos_view_angle_during_observation = observation_phase_indicator * (
            #     detector_cos_view_angle - 1) + 1

            # self.register_output('optics_cos_view_angle_during_observation',
            #                      optics_cos_view_angle_during_observation)
            # self.register_output('detector_cos_view_angle_during_observation',
            #                      detector_cos_view_angle_during_observation)

            # min_optics_cos_view_angle = csdl.min(
            #     optics_cos_view_angle_during_observation)
            # min_detector_cos_view_angle = csdl.min(
            #     detector_cos_view_angle_during_observation)
            # print([op.name
            #        for op in min_optics_cos_view_angle.dependencies])
            # for op in min_optics_cos_view_angle.dependencies:
            #     print([(var.name, var.shape) for var in op.dependencies])
            # print([op.name
            #        for op in min_detector_cos_view_angle.dependencies])
            # for op in min_detector_cos_view_angle.dependencies:
            #     print([(var.name, var.shape) for var in op.dependencies])
            # print('observation_phase_indicator',observation_phase_indicator.shape)
            # print('optics_cos_view_angle',optics_cos_view_angle.shape)
            # print('detector_cos_view_angle',detector_cos_view_angle.shape)
            # print('optics_cos_view_angle_during_observation',optics_cos_view_angle_during_observation.shape)
            # print('detector_cos_view_angle_during_observation',detector_cos_view_angle_during_observation.shape)
            # print('min_optics_cos_view_angle',min_optics_cos_view_angle.shape)
            # print('min_detector_cos_view_angle',min_detector_cos_view_angle.shape)
            # exit()

            # self.register_output('min_optics_cos_view_angle',
            #                      min_optics_cos_view_angle)
            # self.register_output('min_detector_cos_view_angle',
            #                      min_detector_cos_view_angle)
            # self.add_constraint(
            #     'min_optics_cos_view_angle',
            #     lower=np.cos(telescope_view_halfangle_tol_arcsec / deg2arcsec *
            #                  np.pi / 180),
            # )
            # self.add_constraint(
            #     'min_detector_cos_view_angle',
            #     lower=np.cos(telescope_view_halfangle_tol_arcsec / deg2arcsec *
            #                  np.pi / 180),
            # )


if __name__ == '__main__':
    from csdl_om import Simulator
    from lsdo_cubesat.specifications.cubesat_spec import CubesatSpec
    from lsdo_cubesat.specifications.swarm_spec import SwarmSpec

    # initial state relative to reference orbit -- same for all s/c
    initial_orbit_state = np.array([1e-3] * 3 + [1e-3] * 3)

    cubesats = dict()

    cubesats['optics'] = CubesatSpec(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6),
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    )

    cubesats['detector'] = CubesatSpec(
        name='detector',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6),
        specific_impulse=47.,
        perigee_altitude=500.002,
        apogee_altitude=499.98,
    )
    swarm = SwarmSpec(
        num_times=4,
        num_cp=3,
        step_size=95 * 60 / (4 - 1),
        cross_threshold=0.857,
    )
    for v in cubesats.values():
        swarm.add(v)
    rep = GraphRepresentation(TelescopeConfiguration(swarm=swarm, ))
    sim = Simulator(rep)
    sim.visualize_implementation()
