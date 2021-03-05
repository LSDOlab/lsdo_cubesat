import pickle

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns
from PIL import Image

from lsdo_viz.api import BaseViz, Frame

sns.set(style='darkgrid')

earth_radius = 6371.
cubesat_names = ['sunshade', 'optics', 'detector']
ground_station_names = ['UCSD', 'UIUC', 'Georgia', 'Montana']

time = 1501


def XYZ_2_LLA(X, Y, Z):

    ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
    lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

    lon, lat, alt = pyproj.transform(ecef, lla, X, Y, Z, radians=True)

    lon = 180 / np.pi * lon
    lat = 180 / np.pi * lat
    alt = 180 / np.pi * alt

    return lon, lat, alt


def datasort(lon, lat, alt):
    A = np.array([lon, lat, alt])
    B = np.transpose(A)
    C = B[np.lexsort(B[:, ::-1].T)]
    return C


def viz(X, Y, Z):
    lon, lat, alt = XYZ_2_LLA(X, Y, Z)
    matrix = datasort(lon, lat, alt)
    return matrix


def spread_out_orbit(rel_rel_scale, ref_rel_scale, ref, rel, others):
    """
    This function exaggerates the distance between a spacecraft's orbit
    and the reference orbit for the swarm. `ref` and `rel` are n-vectors
    and represent X, Y, or Z component of the position.
    """
    m_to_km = 1e3

    # spread out orbits relative to each other
    drel = rel - rel
    for other in others:
        drel += rel - other
    ex = rel - rel_rel_scale * drel

    # spread out orbits relative to reference orbit
    spread_ref_scale = ref_rel_scale * m_to_km
    exaggerated = ref + spread_ref_scale * ex
    return exaggerated


class Viz(BaseViz):
    def setup(self):
        self.frame_name_format = 'output_{}'

        self.add_frame(
            Frame(
                height_in=20.,
                width_in=25.,
                nrows=6,
                ncols=15,
                wspace=0.4,
                hspace=0.4,
            ), 1)

    def plot(self, data_dict_current, data_dict_all, limits, ind, video=False):
        import matplotlib.image as mpimg
        import numpy as np

        for cubesat_name in cubesat_names:
            reference_orbit = data_dict_current['reference_orbit_state']

            relative_orbit = data_dict_current[
                '{}_cubesat_group.relative_orbit_state'.format(cubesat_name)]

            mask_vec = data_dict_current['mask_vec']

            roll = data_dict_current['{}_cubesat_group.roll'.format(
                cubesat_name)]
            pitch = data_dict_current['{}_cubesat_group.pitch'.format(
                cubesat_name)]

            position = data_dict_current['{}_cubesat_group.position'.format(
                cubesat_name)]
            velocity = data_dict_current['{}_cubesat_group.velocity'.format(
                cubesat_name)]

            thrust_3xn = data_dict_current[
                '{}_cubesat_group.thrust_3xn'.format(cubesat_name)]
            propellant = data_dict_current[
                '{}_cubesat_group.propellant_mass'.format(cubesat_name)]

            mass_flow_rate = data_dict_current[
                '{}_cubesat_group.mass_flow_rate'.format(cubesat_name)]

            data = data_dict_current['{}_cubesat_group.Data'.format(
                cubesat_name)]

            KS_Download_rate = data_dict_current[
                '{}_cubesat_group.KS_Download_rate'.format(cubesat_name)]

            thrust_scalar = data_dict_current[
                '{}_cubesat_group.thrust_scalar'.format(cubesat_name)]

            # number of battery cells
            num_series = data_dict_current[
                '{}_cubesat_group.num_series'.format(cubesat_name)]
            num_series = data_dict_all['{}_cubesat_group.num_series'.format(
                cubesat_name)]

            # propellant mass
            total_propellant_used = data_dict_current[
                '{}_cubesat_group.total_propellant_used'.format(cubesat_name)]
            total_propellant_used = data_dict_all[
                '{}_cubesat_group.total_propellant_used'.format(cubesat_name)]

            # propellant volume
            total_propellant_volume = data_dict_current[
                '{}_cubesat_group.total_propellant_volume'.format(
                    cubesat_name)]
            total_propellant_volume = data_dict_all[
                '{}_cubesat_group.total_propellant_volume'.format(
                    cubesat_name)]

            # battery mass
            battery_mass = data_dict_current[
                '{}_cubesat_group.battery_mass'.format(cubesat_name)]
            battery_mass = data_dict_all[
                '{}_cubesat_group.battery_mass'.format(cubesat_name)]

            # battery volume
            battery_volume = data_dict_current[
                '{}_cubesat_group.battery_volume'.format(cubesat_name)]
            battery_volume = data_dict_all[
                '{}_cubesat_group.battery_volume'.format(cubesat_name)]

            soc = data_dict_current['{}_cubesat_group.cell_model.soc'.format(
                cubesat_name)]

            for ground_station_name in ground_station_names:

                Comm_LOS = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.CommLOS'.format(
                        cubesat_name, ground_station_name)]

                P_comm = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.P_comm'.format(
                        cubesat_name, ground_station_name)]

                data_rate = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.Download_rate'.format(
                        cubesat_name, ground_station_name)]

                distance_to_groundstation = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.distance_to_groundstation'.
                    format(cubesat_name, ground_station_name)]

                r_b2g_I = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.r_b2g_I'.format(
                        cubesat_name, ground_station_name)]

                r_e2g_I = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.r_e2g_I'.format(
                        cubesat_name, ground_station_name)]

                q_E = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.q_E'.format(
                        cubesat_name, ground_station_name)]

                Rot_ECI_EF = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.Rot_ECI_EF'.format(
                        cubesat_name, ground_station_name)]

                r_e2g_E = data_dict_current[
                    '{}_cubesat_group.{}_comm_group.r_e2g_E'.format(
                        cubesat_name, ground_station_name)]

                # orbit_state = data_dict_current[
                #     '{}_cubesat_group.orbit_state_km'.format(
                #         cubesat_name, ground_station_name)]

        normal_distance_sunshade_detector = data_dict_current[
            'normal_distance_sunshade_detector_mm']
        normal_distance_optics_detector = data_dict_current[
            'normal_distance_optics_detector_mm']
        distance_sunshade_optics = data_dict_current[
            'distance_sunshade_optics_mm']
        distance_optics_detector = data_dict_current[
            'distance_optics_detector_mm']
        masked_normal_distance_sunshade_detector = data_dict_current[
            'masked_normal_distance_sunshade_detector_mm_sq_sum']
        masked_normal_distance_optics_detector = data_dict_current[
            'masked_normal_distance_optics_detector_mm_sq_sum']
        masked_distance_sunshade_optics = data_dict_current[
            'masked_distance_sunshade_optics_mm_sq_sum']
        masked_distance_optics_detector = data_dict_current[
            'masked_distance_optics_detector_mm_sq_sum']
        sunshade_relative = data_dict_current[
            'sunshade_cubesat_group.relative_orbit_state_sq_sum']
        optics_relative = data_dict_current[
            'optics_cubesat_group.relative_orbit_state_sq_sum']
        detector_relative = data_dict_current[
            'detector_cubesat_group.relative_orbit_state_sq_sum']
        total_data_download = data_dict_current['total_data_downloaded']
        total_propellant_used = data_dict_current['total_propellant_used']
        observation_dot = data_dict_current['observation_dot']
        obj = data_dict_current['obj']
        obj = data_dict_all['obj']

        # optics_matrix = data_dict_list[ind][
        #     "optics_cubesat_group.relative_orbit_state"][:3, :]
        # r_optics = np.linalg.norm(optics_matrix, ord=1, axis=0)

        # detector_matrix = data_dict_list[ind][
        #     "detector_cubesat_group.relative_orbit_state"][:3, :]
        # r_detector = np.linalg.norm(detector_matrix, ord=1, axis=0)

        # sunshade_matrix = data_dict_list[ind][
        #     "sunshade_cubesat_group.relative_orbit_state"][:3, :]
        # r_sunshade = np.linalg.norm(sunshade_matrix, ord=1, axis=0)
        self.get_frame(1).clear_all_axes()

        with self.get_frame(1)[5, 0:3] as ax:
            # battery_mass = np.array(
            #     data_dict_all['{}_cubesat_group.battery_mass'.format(
            #         cubesat_name)]).flatten()
            # print('battery_mass')
            # print(battery_mass)
            # battery_volume = np.array(
            #     data_dict_all['{}_cubesat_group.battery_volume'.format(
            #         cubesat_name)]).flatten()
            # print('battery_volume')
            # print(battery_volume)
            # total_propellant_used = np.array(
            #     data_dict_all['{}_cubesat_group.total_propellant_used'.format(
            #         cubesat_name)]).flatten()
            # print('total_propellant_used')
            # print(total_propellant_used)
            # total_propellant_volume = np.array(
            #     data_dict_all['{}_cubesat_group.total_propellant_volume'.
            #                   format(cubesat_name)]).flatten()
            # print('total_propellant_volume')
            # print(total_propellant_volume)

            obj = np.array(data_dict_all['obj']).flatten()
            sns.lineplot(x=np.arange(ind), y=obj[:ind], ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['obj'],
                    fig_axis=0,
                    data_axis=0,
                    lower_margin=0.1,
                    upper_margin=0.1,
                    mode='final',
                ))
            ax.set_xlim(0, len(obj))
            ax.set_xlabel('iterations')
            ax.set_ylabel('obj')

        with self.get_frame(1)[5, 4:7] as ax:
            print(data_dict_all.keys())
            num_series = np.array(
                data_dict_all['detector_cubesat_group.num_series']).flatten()
            sns.lineplot(x=np.arange(ind), y=num_series[:ind], ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['detector_cubesat_group.num_series'],
                    fig_axis=0,
                    data_axis=0,
                    lower_margin=0.1,
                    upper_margin=0.1,
                    mode='final',
                ))
            ax.set_xlim(0, len(num_series))
            ax.set_xlabel('iterations')
            ax.set_ylabel('detector_cubesat_group.num_series')

        with self.get_frame(1)[5, 8:10] as ax:
            soc = data_dict_current['detector_cubesat_group.cell_model.soc']
            sns.lineplot(x=np.arange(time).flatten(), y=soc.flatten(), ax=ax)
            # ax.set_ylim(
            #     self.get_limits(
            #         ['detector_cubesat_group.cell_model.soc'],
            #         fig_axis=1,
            #         data_axis=0,
            #         lower_margin=0.1,
            #         upper_margin=0.1,
            #         mode='final',
            #     ))
            ax.set_xlabel('num_times')
            ax.set_ylabel('detector_cubesat_group.cell_model.soc')

        with self.get_frame(1)[0, 0:3] as ax:

            data = data_dict_current['detector_cubesat_group.Data']
            # print(data.shape)

            num_times = np.arange(time)
            # data_rate.reshape((1, 1501))
            data = data.reshape(-1, )
            sns.lineplot(x=num_times, y=data, ax=ax)

            ax.set_ylim(
                self.get_limits(
                    ['detector_cubesat_group.Data'],
                    fig_axis=1,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('Data_downloaded')

        with self.get_frame(1)[1, 0:3] as ax:

            CommLOS = data_dict_current[
                'detector_cubesat_group.UCSD_comm_group.CommLOS']
            # print(data_rate.shape)

            num_times = np.arange(time)
            CommLOS.reshape((1, time))
            sns.lineplot(x=num_times, y=CommLOS, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['detector_cubesat_group.UCSD_comm_group.CommLOS'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('UCSD_CommLOS')

        with self.get_frame(1)[0, 12:15] as ax:

            propellant = data_dict_current[
                '{}_cubesat_group.propellant_mass'.format(cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            sns.lineplot(x=num_times, y=propellant[0, :], ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['{}_cubesat_group.propellant_mass'.format(cubesat_name)],
                    fig_axis=1,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('Propellant')

        with self.get_frame(1)[1, 12:15] as ax:

            mass_flow_rate = data_dict_current[
                '{}_cubesat_group.mass_flow_rate'.format(cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            sns.lineplot(x=num_times, y=mass_flow_rate, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['{}_cubesat_group.mass_flow_rate'.format(cubesat_name)],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('Mass_flow')

        with self.get_frame(1)[0, 4:11] as ax:
            ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
            lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

            reference_orbit = data_dict_current['reference_orbit_state']
            matrix_reference = viz(reference_orbit[0, :],
                                   reference_orbit[1, :],
                                   reference_orbit[2, :])

            path = "/Users/victor/packages/lsdo_cubesat/lsdo_cubesat/map/world.jpg"
            earth = mpimg.imread(path)
            # img = Image.open(path)
            ax.imshow(earth, extent=[-180, 180, -100, 100], aspect='auto')
            ax.plot(matrix_reference[:, 0],
                    matrix_reference[:, 1],
                    linewidth='1',
                    color='yellow')
            # sns.lineplot(x=matrix_reference[:, 0],
            #              y=matrix_reference[:, 1],
            #              linewidth='1',
            #              color='yellow',
            #              ax=ax)
            ax.scatter(-117.2340, 32.8801, marker="p", label="UCSD")
            ax.scatter(-88.2272, 40.1020, marker="p", label="UIUC")
            ax.scatter(-84.3963, 33.7756, marker="p", label="Georgia")
            ax.scatter(-109.533691, 46.9653, marker="p", label="Montana")

            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
            # ax.title("Trajectory of VISORS Satellite")

        with self.get_frame(1)[1, 4:11] as ax:
            ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
            lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

            reference_orbit = data_dict_current['reference_orbit_state']
            matrix_reference = viz(reference_orbit[0, :],
                                   reference_orbit[1, :],
                                   reference_orbit[2, :])

            sns.lineplot(x=matrix_reference[:, 0],
                         y=matrix_reference[:, 2],
                         ax=ax)

            ax.set_xlabel("longitude")
            ax.set_ylabel("altitude")

        with self.get_frame(1)[2, 0:3] as ax:
            num_times = np.arange(time)

            normal_distance_sunshade_detector = data_dict_current[
                'normal_distance_sunshade_detector_mm']

            # print(normal_distance_sunshade_detector.shape)
            normal_distance_sunshade_detector.reshape((1, time))
            sns.lineplot(x=num_times,
                         y=normal_distance_sunshade_detector,
                         ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['normal_distance_sunshade_detector_mm'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            # ax.axvline(1000, color='red')
            # ax.axvline(1400, color='red')
            ax.set_xlabel('num_times')
            ax.set_ylabel('alignment_s_d')

        with self.get_frame(1)[2, 4:7] as ax:
            num_times = np.arange(time)

            normal_distance_optics_detector = data_dict_current[
                'normal_distance_optics_detector_mm']

            normal_distance_optics_detector.reshape((1, time))
            sns.lineplot(x=num_times, y=normal_distance_optics_detector, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['normal_distance_optics_detector_mm'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            # ax.axvline(1000, color='red')
            # ax.axvline(1400, color='red')
            ax.set_xlabel('num_times')
            ax.set_ylabel('alignment_o_d')

        with self.get_frame(1)[2, 8:11] as ax:
            num_times = np.arange(time)

            distance_sunshade_optics = data_dict_current[
                'distance_sunshade_optics_mm']

            distance_sunshade_optics.reshape((1, time))
            sns.lineplot(x=num_times, y=distance_sunshade_optics, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['distance_sunshade_optics_mm'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('seperation_s_o')

        with self.get_frame(1)[2, 12:15] as ax:
            num_times = np.arange(time)

            distance_optics_detector = data_dict_current[
                'distance_optics_detector_mm']

            distance_optics_detector.reshape((1, time))
            sns.lineplot(x=num_times, y=distance_optics_detector, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['distance_optics_detector_mm'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('seperation_o_d')

        with self.get_frame(1)[3, 0:3] as ax:
            num_times = np.arange(time)
            roll = data_dict_current['detector_cubesat_group.roll']
            # print(roll_rate.shape)
            roll.reshape((1, time))
            sns.lineplot(x=num_times, y=roll, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['detector_cubesat_group.roll'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('Roll_scalar')
            # ax.get_xaxis().set_ticks([])

        with self.get_frame(1)[3, 4:7] as ax:
            num_times = np.arange(time)
            pitch = data_dict_current['detector_cubesat_group.pitch']
            # print(pitch.shape)
            pitch.reshape((1, time))
            sns.lineplot(x=num_times, y=pitch, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['detector_cubesat_group.pitch'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('Pitch_scalar')
            # ax.get_xaxis().set_ticks([])

        with self.get_frame(1)[3, 8:11] as ax:

            thrust_scalar = data_dict_current[
                '{}_cubesat_group.thrust_scalar'.format(cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            thrust_scalar.reshape((1, time))
            sns.lineplot(x=num_times, y=thrust_scalar, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['{}_cubesat_group.thrust_scalar'.format(cubesat_name)],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('Thrust_scalar')

        with self.get_frame(1)[3, 12:15] as ax:

            P_comm = data_dict_current[
                '{}_cubesat_group.UCSD_comm_group.P_comm'.format(cubesat_name)]
            # print(data_rate.shape)

            num_times = np.arange(time)
            P_comm.reshape((1, time))
            sns.lineplot(x=num_times, y=P_comm, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    [
                        '{}_cubesat_group.UCSD_comm_group.P_comm'.format(
                            cubesat_name)
                    ],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('P_comm')

        with self.get_frame(1)[4, 0:3] as ax:

            UCSD_data_rate = data_dict_current[
                'detector_cubesat_group.UCSD_comm_group.Download_rate']
            # print(data_rate.shape)

            num_times = np.arange(time)
            sns.lineplot(x=num_times, y=UCSD_data_rate, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['detector_cubesat_group.UCSD_comm_group.Download_rate'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('UCSD_data_rate')

        with self.get_frame(1)[4, 4:7] as ax:

            UIUC_data_rate = data_dict_current[
                'detector_cubesat_group.UIUC_comm_group.Download_rate']
            # print(data_rate.shape)

            num_times = np.arange(time)
            sns.lineplot(x=num_times, y=UIUC_data_rate, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    ['detector_cubesat_group.UIUC_comm_group.Download_rate'],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('UIUC_data_rate')

        with self.get_frame(1)[4, 8:11] as ax:

            Georgia_data_rate = data_dict_current[
                'detector_cubesat_group.Georgia_comm_group.Download_rate']
            # print(Georgia_data_rate)

            num_times = np.arange(time)
            sns.lineplot(x=num_times, y=Georgia_data_rate, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    [
                        'detector_cubesat_group.Georgia_comm_group.Download_rate'
                    ],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('Georgia_data_rate')

        with self.get_frame(1)[4, 12:15] as ax:

            Montana_data_rate = data_dict_current[
                'detector_cubesat_group.Montana_comm_group.Download_rate']
            # print(data_rate.shape)

            num_times = np.arange(time)
            sns.lineplot(x=num_times, y=Montana_data_rate, ax=ax)
            ax.set_ylim(
                self.get_limits(
                    [
                        'detector_cubesat_group.Montana_comm_group.Download_rate'
                    ],
                    fig_axis=0,
                    data_axis=0,
                    mode='final',
                    lower_margin=0.1,
                    upper_margin=0.1,
                ), )
            ax.set_xlabel('num_times')
            ax.set_ylabel('Montana_data_rate')

        self.get_frame(1).write()
