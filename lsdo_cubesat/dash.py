from lsdo_dash.base_dash import BaseDash


class Dash(BaseDash):
    def setup(self):
        # Define what variables to save

        self.set_clientID('simulator', write_stride=3)
        for varname in self.user_varnames:
            self.save_variable(varname, history=True)

        # Define frames
        self.add_frame(1,
                       height_in=8.,
                       width_in=12.,
                       nrows=2,
                       ncols=3,
                       wspace=0.4,
                       hspace=0.4)


if __name__ == '__main__':
    varnames = [
        'times',
        'sun_direction',
        'optics_observation_dot',
        'optics_observation_phase_indicator',
        'detector_observation_dot',
        'detector_observation_phase_indicator',
        'observation_phase_indicator',
        'optics_cubesat_relative_position',
        'detector_cubesat_relative_position',
        'separation_m',
        'separation_error_during_observation',
        'min_separation_error',
        'max_separation_error',
        'view_plane_error_during_observation',
        'min_view_plane_error',
        'max_view_plane_error',
        'telescope_direction',
        'telescope_view_angle_error',
        'max_telescope_view_angle_error',
        'optics_cos_view_angle_error',
        'optics_cos_view_angle_error_during_observation',
        'optics_view_angle_error',
        'max_optics_view_angle_error',
        'detector_cos_view_angle_error',
        'detector_cos_view_angle_error_during_observation',
        'detector_view_angle_error',
        'max_detector_view_angle_error',
        'total_propellant_used',
        'obj',
        'optics_cubesat_group.initial_propellant_mass',
        'optics_cubesat_group.thrust_cp',
        'optics_cubesat_group.thrust_3xn',
        'optics_cubesat_group.mass_flow_rate',
        'optics_cubesat_group.propellant_mass',
        'optics_cubesat_group.total_propellant_used',
        'optics_cubesat_group.total_propellant_volume',
        'optics_cubesat_group.dry_mass',
        'optics_cubesat_group.initial_orbit_state',
        'optics_cubesat_group.mass',
        'optics_cubesat_group.radius_m',
        'optics_cubesat_group.relative_orbit_state_m',
        'optics_cubesat_group.orbit_state_km',
        'optics_cubesat_group.position_km',
        'optics_cubesat_group.velocity_km_s',
        'optics_cubesat_group.ks_altitude_km',
        'optics_cubesat_group.RTN_from_ECI',
        'optics_cubesat_group.osculating_orbit_angular_speed',
        'optics_cubesat_group.yaw_cp',
        'optics_cubesat_group.pitch_cp',
        'optics_cubesat_group.roll_cp',
        'optics_cubesat_group.initial_rw_speed',
        'optics_cubesat_group.pitch',
        'optics_cubesat_group.yaw',
        'optics_cubesat_group.roll',
        'optics_cubesat_group.B_from_ECI',
        'optics_cubesat_group.B_from_ECI_dot',
        'optics_cubesat_group.body_rates',
        'optics_cubesat_group.body_accels',
        'optics_cubesat_group.bt1',
        'optics_cubesat_group.B_from_RTN',
        'optics_cubesat_group.gravity_term',
        'optics_cubesat_group.body_torque',
        'optics_cubesat_group.rw_speed',
        'optics_cubesat_group.rw_accel',
        'optics_cubesat_group.rw_torque',
        'optics_cubesat_group.rw_speed_ks_min',
        'optics_cubesat_group.rw_speed_ks_max',
        'optics_cubesat_group.rw_power',
        'optics_cubesat_group.sun_LOS',
        'optics_cubesat_group.sun_component',
        'optics_cubesat_group.percent_exposed_area',
        'optics_cubesat_group.load_current',
        'optics_cubesat_group.load_voltage',
        'optics_cubesat_group.solar_power',
        'optics_cubesat_group.battery_power',
        'optics_cubesat_group.num_series',
        'optics_cubesat_group.num_parallel',
        'optics_cubesat_group.power',
        'optics_cubesat_group.battery_mass',
        'optics_cubesat_group.battery_volume',
        'optics_cubesat_group.initial_soc',
        'optics_cubesat_group.min_soc',
        'optics_cubesat_group.max_soc',
        'optics_cubesat_group.delta_soc',
        'optics_cubesat_group.min_current',
        'optics_cubesat_group.max_current',
        'optics_cubesat_group.battery_and_propellant_volume',
        'detector_cubesat_group.initial_propellant_mass',
        'detector_cubesat_group.thrust_cp',
        'detector_cubesat_group.thrust_3xn',
        'detector_cubesat_group.mass_flow_rate',
        'detector_cubesat_group.propellant_mass',
        'detector_cubesat_group.total_propellant_used',
        'detector_cubesat_group.total_propellant_volume',
        'detector_cubesat_group.dry_mass',
        'detector_cubesat_group.initial_orbit_state',
        'detector_cubesat_group.mass',
        'detector_cubesat_group.radius_m',
        'detector_cubesat_group.relative_orbit_state_m',
        'detector_cubesat_group.orbit_state_km',
        'detector_cubesat_group.position_km',
        'detector_cubesat_group.velocity_km_s',
        'detector_cubesat_group.ks_altitude_km',
        'detector_cubesat_group.RTN_from_ECI',
        'detector_cubesat_group.osculating_orbit_angular_speed',
        'detector_cubesat_group.yaw_cp',
        'detector_cubesat_group.pitch_cp',
        'detector_cubesat_group.roll_cp',
        'detector_cubesat_group.initial_rw_speed',
        'detector_cubesat_group.pitch',
        'detector_cubesat_group.yaw',
        'detector_cubesat_group.roll',
        'detector_cubesat_group.B_from_ECI',
        'detector_cubesat_group.B_from_ECI_dot',
        'detector_cubesat_group.body_rates',
        'detector_cubesat_group.body_accels',
        'detector_cubesat_group.bt1',
        'detector_cubesat_group.B_from_RTN',
        'detector_cubesat_group.gravity_term',
        'detector_cubesat_group.body_torque',
        'detector_cubesat_group.rw_speed',
        'detector_cubesat_group.rw_accel',
        'detector_cubesat_group.rw_torque',
        'detector_cubesat_group.rw_speed_ks_min',
        'detector_cubesat_group.rw_speed_ks_max',
        'detector_cubesat_group.rw_power',
        'detector_cubesat_group.sun_LOS',
        'detector_cubesat_group.sun_component',
        'detector_cubesat_group.percent_exposed_area',
        'detector_cubesat_group.load_current',
        'detector_cubesat_group.load_voltage',
        'detector_cubesat_group.solar_power',
        'detector_cubesat_group.battery_power',
        'detector_cubesat_group.num_series',
        'detector_cubesat_group.num_parallel',
        'detector_cubesat_group.power',
        'detector_cubesat_group.battery_mass',
        'detector_cubesat_group.battery_volume',
        'detector_cubesat_group.initial_soc',
        'detector_cubesat_group.min_soc',
        'detector_cubesat_group.max_soc',
        'detector_cubesat_group.delta_soc',
        'detector_cubesat_group.min_current',
        'detector_cubesat_group.max_current',
        'detector_cubesat_group.battery_and_propellant_volume',
        'optics_cubesat_group.voltage',
        'optics_cubesat_group.soc',
        'detector_cubesat_group.voltage',
        'detector_cubesat_group.soc',
    ]
    d = Dash(varnames)
    d.run_GUI()
