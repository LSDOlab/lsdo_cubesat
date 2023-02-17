from csdl import Model, Output
import csdl
import numpy as np

# from lsdo_cubesat.cubesat_group import Cubesat
from lsdo_cubesat.configurations.virtual_telescope import VirtualTelescope
from lsdo_cubesat.specifications.swarm_spec import SwarmSpec
from lsdo_cubesat.configurations.cubesat import Cubesat
from lsdo_cubesat.disciplines.sun.sun_direction import SunDirection
from lsdo_cubesat.constants import s


class VirtualTelescope(Model):

    def register_connected(self, prefix: str, name: str,
                           var: Output) -> Output:
        self.register_output(prefix + name, var)
        return var

    def import_vars(self, name, shape=(1, )):
        optics = self.declare_variable(f'optics_{name}', shape=shape)
        detector = self.declare_variable(f'detector_{name}', shape=shape)

        self.connect_vars(name)

        return optics, detector

    def connect_vars(self, name):
        self.connect(
            f'optics_cubesat.{name}',
            f'optics_{name}',
        )
        self.connect(
            f'detector_cubesat.{name}',
            f'detector_{name}',
        )

    def initialize(self):
        self.parameters.declare('swarm', types=SwarmSpec)
        self.parameters.declare('duration', types=float)
        self.parameters.declare('telescope_length_tol_mm',
                                default=15.,
                                types=float)
        self.parameters.declare('telescope_view_plane_tol_mm',
                                default=18.,
                                types=float)

    def define(self):
        swarm = self.parameters['swarm']

        num_times = swarm['num_times']
        num_cp = swarm['num_cp']
        step_size = swarm['step_size']
        duration = swarm['duration']

        telescope_length_tol_mm = self.parameters['telescope_length_tol_mm']
        telescope_view_plane_tol_mm = self.parameters[
            'telescope_view_plane_tol_mm']

        # TODO: What are we modeling here?
        earth_orbit_angular_speed_rad_min = 2 * np.pi / 365 * 1 / 24 * 1 / 60
        step_size_min = duration / num_times
        earth_orbit_angular_position = earth_orbit_angular_speed_rad_min * step_size_min * np.arange(
            num_times)
        v = np.zeros((num_times, 3))
        v[:, 0] = 1
        # v[:, 0] = np.cos(earth_orbit_angular_position)
        # v[:, 1] = np.sin(earth_orbit_angular_position)

        h = np.ones(num_times - 1) * step_size
        self.create_input('h', val=h)
        # sun_direction (in ECI frame) used in TelescopeConfiguration;
        # here it is an input because it is precomputed
        sun_direction = self.create_input(
            'sun_direction',
            shape=(num_times, 3),
            val=v,
        )

        # add model for each cubesat in telescope (2 total; optics and
        # detector)
        for cubesat in swarm.children:
            cubesat_name = cubesat['name']
            submodel_name = '{}_cubesat'.format(cubesat_name)
            self.add(
                Cubesat(
                    num_times=num_times,
                    num_cp=num_cp,
                    step_size=step_size,
                    cubesat=cubesat,
                    # mtx=get_bspline_mtx(num_cp, num_times, order=4),
                ),
                name='{}_cubesat'.format(cubesat_name),
                # promotes=['reference_orbit_state_km'],
                promotes=[],
            )
            self.connect('sun_direction',
                         '{}.sun_direction'.format(submodel_name))

        # add constraints for defining telescope configuration
        self.add(
            VirtualTelescope(swarm=swarm),
            name='telescope_config',
        )
        self.connect_vars('sun_LOS')
        # self.connect_vars('relative_orbit_state_m')
        # TODO: connect
        # self.connect_vars('orbit_state')
        # self.connect_vars('B_from_ECI')

        # Define Objective

        ## acceleration_due_to_thrust
        # optics_acceleration_due_to_thrust, detector_acceleration_due_to_thrust = self.import_vars(
        #     'acceleration_due_to_thrust', shape=(num_times, 3))
        # a = optics_acceleration_due_to_thrust + detector_acceleration_due_to_thrust
        # total_acceleration_due_to_thrust = csdl.sum(a*csdl.tanh(5*a))
        # self.register_output('total_acceleration_due_to_thrust', total_acceleration_due_to_thrust)
        # obj = 10*total_acceleration_due_to_thrust

        ## total_propellant_used
        # optics_total_propellant_used, detector_total_propellant_used = self.import_vars(
        #     'total_propellant_used')
        # total_propellant_used = optics_total_propellant_used + detector_total_propellant_used
        # self.register_output('total_propellant_used', total_propellant_used)

        ## total_propellant_mass using initial propellant mass
        # optics_initial_propellant_mass, detector_initial_propellant_mass = self.import_vars(
        #     'initial_propellant_mass')
        # total_propellant_mass = optics_initial_propellant_mass + detector_initial_propellant_mass
        # self.register_output('total_propellant_mass', total_propellant_mass)

        ## penalties
        max_separation_error_during_observation = self.declare_variable(
            'max_separation_error_during_observation')
        max_view_plane_error = self.declare_variable('max_view_plane_error')
        max_telescope_view_angle = self.declare_variable(
            'max_telescope_view_angle')

        # obj = 10*total_propellant_mass + (max_separation_error_during_observation - (
        #     telescope_length_tol_mm / 1000.)**2 + max_view_plane_error - (
        #         telescope_view_plane_tol_mm / 1000.)**2)

        # use 10 coefficient to scale objective to be ~1
        # obj = 10*total_propellant_used
        # obj = ((
        #     max_separation_error_during_observation -
        #     (telescope_length_tol_mm / 1000.)**2 + max_view_plane_error -
        #     (telescope_view_plane_tol_mm / 1000.)**2 + 1e8*max_telescope_view_angle))
        obj = 1.001 * max_telescope_view_angle
        # obj = 10*total_propellant_used + ((
        #     max_separation_error_during_observation -
        #     (telescope_length_tol_mm / 1000.)**2 + max_view_plane_error -
        #     (telescope_view_plane_tol_mm / 1000.)**2))/s
        # obj = total_propellant_used + 100*((
        #     max_separation_error_during_observation -
        #     (telescope_length_tol_mm / 1000.)**2 + max_view_plane_error -
        #     (telescope_view_plane_tol_mm / 1000.)**2))
        # obj = total_acceleration_due_to_thrust + (
        #     max_separation_error_during_observation -
        #     (telescope_length_tol_mm / 1000.)**2 + max_view_plane_error -
        #     (telescope_view_plane_tol_mm / 1000.)**2)
        # obj = (max_separation_error_during_observation -
        #        (telescope_length_tol_mm / 1000.)**2 + max_view_plane_error -
        #        (telescope_view_plane_tol_mm / 1000.)**2)
        # obj = 10 * max_separation_error_during_observation
        # obj= (csdl.sum(separation_error_during_observation) +
        #        csdl.sum(view_plane_error_during_observation) - num_times *
        #        ((telescope_length_tol_mm / 1000.)**2 +
        #         (telescope_view_plane_tol_mm / 1000.)**2))
        self.register_output('obj', obj)
        self.add_objective('obj')
