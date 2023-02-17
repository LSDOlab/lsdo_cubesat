import lsdo_dash.api as ld
import seaborn as sns
import numpy as np
from typing import Tuple, List, Union

from matplotlib import rc

rc('text', usetex=True)

sns.set()
"""
This script creates the Dash class which defines
- plotting procedure
- which variables to save
- configuration

Run this script independently of optimziation procedure.

Note that specified variable values are saved every iteration externally in "case_archive".
It is recommended to occasionally delete data from unneeded optimization runs to prevent storage
of unnecessary data.
"""

# TODO: make plot function specifically for indicating observation phase
# using a color coded interval


def save_variables(dash: ld.BaseDash, variables: List[str], *, history: bool):
    for variable in variables:
        dash.save_variable(variable, history=history)


def plot_historical(
    frame,
    data_dict,
    subplot: Tuple[int, int],
    varname: str,
    plot_title: str,
    ylabel: str,
):

    ax_subplot = frame[subplot]
    x_axis = data_dict['simulator']['global_ind']
    y_axis = data_dict['simulator'][varname]
    sns.lineplot(
        x=x_axis,
        y=np.array(y_axis).flatten(),
        ax=ax_subplot,
    )
    ax_subplot.set_ylabel(ylabel)
    ax_subplot.set_xlabel('Iteration')
    ax_subplot.set_title(plot_title)
    return ax_subplot


def plot(
    frame,
    data_dict,
    subplot: Tuple[int, int],
    varname: str,
    plot_title: str,
    ylabel: str,
    xlabel: str,
):
    ax_subplot = frame[subplot]
    y_axis = data_dict['simulator'][varname]

    if data_indices is None:
        sns.lineplot(
            x=x_axis,
            y=np.array(y_axis).flatten(),
            ax=ax_subplot,
        )
    else:
        for index in data_indices:
            sns.lineplot(
                x=x_axis,
                y=np.array(y_axis[:, index]).flatten(),
                ax=ax_subplot,
            )
    ax_subplot.set_ylabel(ylabel)
    ax_subplot.set_xlabel(xlabel)
    ax_subplot.set_title(plot_title)


def coplot_historical(
    frame,
    data_dict,
    subplot: Tuple[int, int],
    varnames: List[str],
    plot_title: str,
    ylabel: str,
):
    ax_subplot = frame[subplot]
    x_axis = data_dict['simulator']['global_ind']
    for varname in varnames:
        y_axis = data_dict['simulator'][varname]
        sns.lineplot(
            x=x_axis,
            y=np.array(y_axis).flatten(),
            ax=ax_subplot,
        )
    ax_subplot.set_ylabel(ylabel)
    ax_subplot.set_xlabel('Iteration')
    ax_subplot.set_title(plot_title)


def coplot(
    frame,
    data_dict,
    subplot: Tuple[int, int],
    varnames: List[str],
    plot_title: str,
    ylabel: str,
    xlabel: str,
    data_indices: List[Union[List[int], None]] = [],
):
    global A
    x_axis = np.concatenate(
        (np.array([0]), np.matmul(A, data_dict['simulator']['h'])))
    ax_subplot = frame[subplot]
    if data_indices == []:
        for varname in varnames:
            y_axis = data_dict['simulator'][varname]
            sns.lineplot(
                x=x_axis,
                y=np.array(y_axis).flatten(),
                ax=ax_subplot,
            )
    else:
        for (indices, varname) in zip(data_indices, varnames):
            y_axis = data_dict['simulator'][varname]
            if indices is None:
                sns.lineplot(
                    x=x_axis,
                    y=np.array(y_axis).flatten(),
                    ax=ax_subplot,
                )
            else:
                for index in indices:
                    sns.lineplot(
                        x=x_axis,
                        y=np.array(y_axis[:, index]).flatten(),
                        ax=ax_subplot,
                    )
    ax_subplot.set_ylabel(ylabel)
    ax_subplot.set_xlabel(xlabel)
    ax_subplot.set_title(plot_title)
    return ax_subplot


class Dash(ld.BaseDash):

    def setup(self):
        # Define what variables to save

        # Add variable from simulator client
        self.set_clientID('simulator')
        # name of csdl variable
        # history = True when plotting values of previous optimization
        # iterations

        # Save historical data for scalar values only, otherwise
        # difficult to visualize history; include objective,
        # constraints, and design variables
        save_variables(
            self,
            [
                'obj',
                # 'total_propellant_used',
                'min_separation_during_observation',
                'max_separation_during_observation',
                'optics_cubesat.acceleration_due_to_thrust',
                'detector_cubesat.acceleration_due_to_thrust',
                # 'optics_cubesat.initial_propellant_mass',
                # 'detector_cubesat.initial_propellant_mass',
                # 'min_telescope_view_angle',
                'max_telescope_view_angle',
                # 'optics_cubesat.min_reaction_wheel_torque',
                # 'detector_cubesat.min_reaction_wheel_torque',
                # 'optics_cubesat.max_reaction_wheel_torque',
                # 'detector_cubesat.max_reaction_wheel_torque',
            ],
            history=True,
        )

        # Save data for time dependent variables; include constraints,
        # design variables, and all intermediate variables that store
        # trajectories;
        # if video is True, store data from all iterations to make video
        # showing evolution of time dependent variables across
        # iterations
        video = True
        save_variables(
            self,
            [
                'h',
            ],
            history=video,
        )
        # Telescope Info (Translational)
        save_variables(
            self,
            [
                'separation_m',
                'observation_phase_indicator',
                'view_plane_error',
                # 'telescope_cos_view_angle',
                'telescope_view_angle',
                'telescope_view_angle_unmasked',
            ],
            history=video,
        )
        # Telescope Info (Rotational)
        # save_variables(
        # self,
        # [
        # 'optics_cos_view_angle',
        # 'detector_cos_view_angle',
        # 'optics_cos_view_angle_during_observation',
        # 'detector_cos_view_angle_during_observation',
        # 'optics_cubesat.B_from_ECI',
        # 'detector_cubesat.B_from_ECI',
        # ],
        # history=video,
        # )

        # Translational Info
        save_variables(
            self,
            [
                # 'optics_cubesat.thrust_cp',
                # 'detector_cubesat.thrust_cp',
                # 'optics_cubesat.thrust',
                # 'detector_cubesat.thrust',
                'optics_cubesat.acceleration_due_to_thrust',
                'detector_cubesat.acceleration_due_to_thrust',
                'optics_cubesat.orbit_state',
                'detector_cubesat.orbit_state',
                # 'optics_cubesat.relative_orbit_state_m',
                # 'detector_cubesat.relative_orbit_state_m',
                # 'optics_cubesat.propellant_mass',
                # 'detector_cubesat.propellant_mass',
            ],
            history=video,
        )

        # Rotational Info
        # save_variables(
        #     self,
        #     [
        #         'optics_cubesat.yaw_cp',
        #         'optics_cubesat.pitch_cp',
        #         'optics_cubesat.roll_cp',
        #         'optics_cubesat.yaw',
        #         'optics_cubesat.pitch',
        #         'optics_cubesat.roll',
        #         'detector_cubesat.yaw_cp',
        #         'detector_cubesat.pitch_cp',
        #         'detector_cubesat.roll_cp',
        #         'detector_cubesat.yaw',
        #         'detector_cubesat.pitch',
        #         'detector_cubesat.roll',
        #         'optics_cubesat.body_rates',
        #         'detector_cubesat.body_rates',
        #         'optics_cubesat.body_torque',
        #         'detector_cubesat.body_torque',
        #         'optics_cubesat.initial_reaction_wheel_velocity',
        #         'detector_cubesat.initial_reaction_wheel_velocity',
        #         'optics_cubesat.reaction_wheel_velocity',
        #         'detector_cubesat.reaction_wheel_velocity',
        #         'optics_cubesat.rw_accel',
        #         'detector_cubesat.rw_accel',
        #         'optics_cubesat.reaction_wheel_torque',
        #         'detector_cubesat.reaction_wheel_torque',
        #         'optics_cubesat.osculating_orbit_angular_velocity',
        #         'detector_cubesat.osculating_orbit_angular_velocity',
        #         # 'optics_cubesat.rw_speed_min',
        #         # 'detector_cubesat.rw_speed_min',
        #         # 'optics_cubesat.rw_torque_min',
        #         # 'detector_cubesat.rw_torque_min',
        #         # 'optics_cubesat.rw_speed_max',
        #         # 'detector_cubesat.rw_speed_max',
        #         # 'optics_cubesat.rw_torque_max',
        #         # 'detector_cubesat.rw_torque_max',
        #     ],
        #     history=video,
        # )

        # Energy Info

        # Define frames
        self.add_frame(
            'Optimization Problem',
            height_in=8.,
            width_in=12.,
            ncols=1,
            nrows=5,
            wspace=0.4,
            hspace=0.4,
        )
        # # Define frames
        # self.add_frame(
        #     'Optimization Problem 2',
        #     height_in=8.,
        #     width_in=12.,
        #     ncols=2,
        #     nrows=2,
        #     wspace=0.4,
        #     hspace=0.4,
        # )
        self.add_frame(
            'Telescope Constraints',
            height_in=8.,
            width_in=12.,
            ncols=1,
            nrows=4,
            wspace=0.4,
            hspace=0.4,
        )
        self.add_frame(
            'Translational Dynamics',
            height_in=8.,
            width_in=12.,
            ncols=2,
            nrows=3,
            wspace=0.4,
            hspace=0.4,
        )
        # self.add_frame(
        #     'Attitude Dynamics',
        #     height_in=8.,
        #     width_in=12.,
        #     ncols=2,
        #     nrows=3,
        #     wspace=0.4,
        #     hspace=0.4,
        # )

    def plot(self,
             frames,
             data_dict_current,
             data_dict_history,
             limits_dict,
             video=False):
        """
        Gets called to create a frame given optimization output. Defined by user.
        The method should have the following high level structure:
        - Read variables from argument
        - call "clear_frame" method for each frame
        - plot variables onto each frame
        - call "save_frame" method for each frame

        Parameters
        ----------
            frames: Frame
                Frame object defined in setup
            data_dict_current: dictionary
                dictionary where keys (names of variable) contain current iteration values of respective variable
            data_dict_all: dictionary
                dictionary where keys (names of variable) contain all iteration values of respective variable
        """

        # data_dict_history has the following key-value structure:
        # data_dict_history[client ID][variable name] = value
        # data_dict_history[client ID]['global ind'] = list of global index

        # example, if there is data saved from two clients in the following order:
        # ozone, ozone, optimizer, ozone, ozone, optimizer,
        # data_dict_history['ozone']['global ind']     = [0, 1, 3, 4]

        # To plot, get frame from argument

        # ax_subplot = frame[0, 0]
        # x_axis = data_dict_history['simulator']['global_ind']
        # cur_ind = x_axis[-1]
        # y_axis = data_dict_history['simulator']['obj'][:cur_ind]
        # sns.lineplot(x=x_axis, y=np.array(y_axis).flatten(), ax=ax_subplot)
        # ax_subplot.set_ylabel('opt objective')
        # ax_subplot.set_xlabel('iteration')
        # ax_subplot.set_title('objective history')

        frame = frames['Optimization Problem']
        frame.clear_all_axes()

        ax = plot_historical(
            frame,
            data_dict_history,
            (0, 0),
            'obj',
            'Objective',
            'Objective',
        )
        # ax.set_ylim((0, 20000))

        ax = plot_historical(
            frame,
            data_dict_history,
            (1, 0),
            'max_separation_error_during_observation',
            'Maximum Separation Error During Observation',
            'Maximum Separation [$m^2$]',
        )
        # ax.set_ylim((0, 2))

        ax = plot_historical(
            frame,
            data_dict_history,
            (2, 0),
            'max_view_plane_error',
            'Maximum View Plane Error During Observation',
            'Maximum View Plane Error [$m^2$]',
        )
        # ax = plot_historical(
        #     frame,
        #     data_dict_history,
        #     (3, 0),
        #     'min_telescope_view_angle',
        #     'Minimum Telescope View Angle During Observation',
        #     'View Angle [rad]',
        # )
        ax = plot_historical(
            frame,
            data_dict_history,
            (4, 0),
            'max_telescope_view_angle',
            'Maximum Telescope View Angle During Observation',
            'View Angle [arcsec]',
        )
        # ax.set_ylim((0, 2))

        # ax = plot_historical(
        #     frame,
        #     data_dict_history,
        #     (3, 0),
        #     'optics_cubesat.initial_propellant_mass',
        #     'Optics Propellant Mass',
        #     'Propellant Mass [kg]',
        # )
        # ax = plot_historical(
        #     frame,
        #     data_dict_history,
        #     (4, 0),
        #     'detector_cubesat.initial_propellant_mass',
        #     'Detector Propellant Mass',
        #     'Propellant Mass [kg]',
        # )

        frame.write()

        # frame = frames['Optimization Problem 2']
        # frame.clear_all_axes()

        # ax = plot_historical(
        #     frame,
        #     data_dict_history,
        #     (0, 0),
        #     'optics_cubesat.min_reaction_wheel_torque',
        #     'Optics Min Reacion Wheel Torque',
        #     'Torque [mN-m]',
        # )
        # ax = plot_historical(
        #     frame,
        #     data_dict_history,
        #     (1, 0),
        #     'optics_cubesat.max_reaction_wheel_torque',
        #     'Optics Max Reacion Wheel Torque',
        #     'Torque [mN-m]',
        # )
        # ax = plot_historical(
        #     frame,
        #     data_dict_history,
        #     (0, 1),
        #     'detector_cubesat.min_reaction_wheel_torque',
        #     'Detector Min Reacion Wheel Torque',
        #     'Torque [mN-m]',
        # )
        # ax = plot_historical(
        #     frame,
        #     data_dict_history,
        #     (1, 1),
        #     'detector_cubesat.max_reaction_wheel_torque',
        #     'Detector Max Reacion Wheel Torque',
        #     'Torque [mN-m]',
        # )
        # frame.write()

        frame = frames['Translational Dynamics']
        frame.clear_all_axes()

        global A

        n = len(data_dict_current['simulator']['h'])
        A = np.tril(np.ones((n, n)), k=-1)

        ax = coplot(
            frame,
            data_dict_current,
            (0, 0),
            [
                # 'optics_cubesat.thrust',
                'optics_cubesat.acceleration_due_to_thrust',
            ],
            'Optics Thrust Profile',
            'Thrust',
            'Time',
            data_indices=[
                [0, 1, 2],
            ],
        )
        # ax.set_ylim((-5e-5, 5e-5))

        ax = coplot(
            frame,
            data_dict_current,
            (0, 1),
            [
                'detector_cubesat.acceleration_due_to_thrust',
            ],
            'Detector Thrust Profile',
            'Thrust',
            'Time',
            data_indices=[
                [0, 1, 2],
            ],
        )
        # ax.set_ylim((-5e-5, 5e-5))

        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (1, 0),
        #     [
        #         'optics_cubesat.propellant_mass',
        #     ],
        #     'Optics Propellant Mass',
        #     'Propellant Mass',
        #     'Time',
        # )
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (1, 1),
        #     [
        #         'detector_cubesat.propellant_mass',
        #     ],
        #     'Detector Propellant Mass',
        #     'Propellant Mass',
        #     'Time',
        # )

        ax = coplot(
            frame,
            data_dict_current,
            (2, 0),
            [
                'observation_phase_indicator',
            ],
            'Observation Phase Indicator',
            'State',
            'Time',
        )
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (1, 0),
        #     [
        #         'optics_cubesat.osculating_orbit_angular_velocity',
        #     ],
        #     'Osculating Orbit Angular Velocity',
        #     'Angular Velocity [1/s?]',
        #     'Time',
        #     data_indices=[
        #         [0, 1, 2],
        #     ],
        # )
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (1, 1),
        #     [
        #         'detector_cubesat.osculating_orbit_angular_velocity',
        #     ],
        #     'Osculating Orbit Angular Velocity',
        #     'Angular Velocity [1/s?]',
        #     'Time',
        #     data_indices=[
        #         [0, 1, 2],
        #     ],
        # )
        frame.write()

        frame = frames['Telescope Constraints']
        frame.clear_all_axes()

        ax = coplot(
            frame,
            data_dict_current,
            (0, 0),
            [
                'separation_error',
            ],
            'Separation Error',
            'Separation Error [$m^2$]',
            'Time',
        )
        # ax.set_ylim((0, 0.01))

        ax = coplot(
            frame,
            data_dict_current,
            (1, 0),
            [
                'view_plane_error',
            ],
            'View Plane Error',
            'View Plane Error [$m^2$]',
            'Time',
        )
        # ax.set_ylim((0, 0.01))
        ax = coplot(
            frame,
            data_dict_current,
            (2, 0),
            [
                'telescope_view_angle',
            ],
            'Telescope View Angle',
            'View Angle [arcsec]',
            'Time',
        )
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (2, 0),
        #     [
        #         # 'optics_cos_view_angle',
        #         # 'detector_cos_view_angle',
        #         'optics_cos_view_angle_during_observation',
        #         'detector_cos_view_angle_during_observation',
        #     ],
        #     'Cosine of View Angle for Each CubeSat',
        #     'Cosine of View Angle [--]',
        #     'Time',
        # )

        ax = coplot(
            frame,
            data_dict_current,
            (3, 0),
            [
                'observation_phase_indicator',
            ],
            'Observation Phase Indicator',
            'Indicator [--]',
            'Time',
        )
        frame.write()

        # frame = frames['Attitude Dynamics']
        # frame.clear_all_axes()

        # # euler angles
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (0, 0),
        #     [
        #         'optics_cubesat.yaw',
        #         'optics_cubesat.pitch',
        #         'optics_cubesat.roll',
        #     ],
        #     'Optics CubeSat Orientation',
        #     'Euler Angle [rad]',
        #     'Time',
        # )
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (0, 1),
        #     [
        #         'detector_cubesat.yaw',
        #         'detector_cubesat.pitch',
        #         'detector_cubesat.roll',
        #     ],
        #     'Detector CubeSat Orientation',
        #     'Euler Angle [rad]',
        #     'Time',
        # )

        # # body angular velocity
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (1, 0),
        #     [
        #         'optics_cubesat.body_rates',
        #     ],
        #     'Optics CubeSat Angular Velocities',
        #     'Angular Velocity [rad/s]',
        #     'Time',
        #     data_indices=[
        #         [0, 1, 2],
        #     ],
        # )
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (1, 1),
        #     [
        #         'detector_cubesat.body_rates',
        #     ],
        #     'Detector CubeSat Angular Velocities',
        #     'Angular Velocity [rad/s]',
        #     'Time',
        #     data_indices=[
        #         [0, 1, 2],
        #     ],
        # )

        # ax_subplot = frame[2,0]
        # y_axis = -data_dict_current['simulator']['optics_cubesat.B_from_ECI'][1,0,:]
        # x_axis = np.arange(len(y_axis)) * 95
        # sns.lineplot(
        #     x=x_axis,
        #     y=np.array(y_axis).flatten(),
        #     ax=ax_subplot,
        # )
        # y_axis = data_dict_current['simulator']['optics_cos_view_angle']
        # sns.lineplot(
        #     x=x_axis,
        #     y=np.array(y_axis).flatten(),
        #     ax=ax_subplot,
        # )
        # ax_subplot.set_ylabel('C')
        # ax_subplot.set_xlabel('Time')
        # ax_subplot.set_title('TEST')

        # ax_subplot = frame[2,1]
        # y_axis = -data_dict_current['simulator']['detector_cubesat.B_from_ECI'][0,0,:]
        # sns.lineplot(
        #     x=x_axis,
        #     y=np.array(y_axis).flatten(),
        #     ax=ax_subplot,
        # )
        # y_axis = data_dict_current['simulator']['detector_cos_view_angle']
        # sns.lineplot(
        #     x=x_axis,
        #     y=np.array(y_axis).flatten(),
        #     ax=ax_subplot,
        # )
        # ax_subplot.set_ylabel('C')
        # ax_subplot.set_xlabel('Time')
        # ax_subplot.set_title('TEST')

        # # reaction wheel angular velocity
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (2, 0),
        #     [
        #         'optics_cubesat.reaction_wheel_velocity',
        #     ],
        #     'Optics CubeSat Reaction Wheel Angular Velocities',
        #     'Angular Velocity [rad/s]',
        #     'Time',
        #     data_indices=[
        #         [0, 1, 2],
        #     ],
        # )
        # ax = coplot(
        #     frame,
        #     data_dict_current,
        #     (2, 1),
        #     [
        #         'detector_cubesat.reaction_wheel_velocity',
        #     ],
        #     'Detector CubeSat Reaction Wheel Angular Velocities',
        #     'Angular Velocity [rad/s]',
        #     'Time',
        #     data_indices=[
        #         [0, 1, 2],
        #     ],
        # )

        # # Write the frame
        # frame.write()


if __name__ == '__main__':
    # Run this script independent of optimization
    # Find images in directory:
    """

    directory:
    "case_archive"
     |- "<time of optimization>"
         |- "_frames" (for images)
             |- frame1.png
             .
             .
             .
             |- framen.png
         |- "_data" (for data)
             |- <values recorded time 1>.png
             .
             .
             .
             |- <values recorded time n>.png
    """

    dash_object = Dash()

    # uncomment to produce image for final frame
    dash_object.visualize_most_recent(show=True)

    # uncomment to produces images for all frames
    # dash_object.visualize(show=True)

    # uncomment to produces images for n_th frame
    # n = 0
    # dash_object.visualize(frame_ind=n, show=True)

    # uncomment to make movie
    # dash_object.visualize_all()
    # dash_object.make_mov()

    # uncomment to run gui
    # dash_object.run_GUI()
