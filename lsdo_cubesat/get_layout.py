from lsdo_viz.api import Div, Animation, Slider, Tabs, Plot2D, Plot3D, generate_vector_func, generate_iopt_func, generate_earth_stl_func


def generate_vector_func_scaled_sum(var1_name, var2_name, scale, indices_func,
                                    ind):
    def vector_func(om, dep):
        array1 = om.get(var1_name)
        array2 = om.get(var2_name)
        array = array1 + scale * array2
        indices = indices_func(om, dep)

        return array[indices, ind, :]

    return vector_func


def generate_vector_func_e(var_name, indices_func):
    def vector_func(om, dep):
        array = om.get(var_name)
        indices = indices_func(om, dep)

        return array[indices, :, 0]

    return vector_func


def generate_vector_func_ind(var_name, indices_func, ind):
    def vector_func(om, dep):
        array = om.get(var_name)
        indices = indices_func(om, dep)

        return array[indices, ind]

    return vector_func


def generate_vector_func_e_rev(var_name, indices_func):
    def vector_func(om, dep):
        array = om.get(var_name)
        indices = indices_func(om, dep)

        return array[indices, :, 0][::-1]

    return vector_func


def transpose_list(old_list, n1):
    assert len(old_list) % n1 == 0
    n2 = int(len(old_list) / n1)

    new_list = []
    for i1 in range(n1):
        for i2 in range(n2):
            new_list.append(old_list[i2 * n1 + i1])

    return new_list


def get_layout(prob):
    layout = Div()

    row = Animation(num_columns=3, default_frame_time=0.1)
    layout.add_element(row)

    slider = Slider('hist', interval_id='interval', stride_id='stride')
    layout.add_element(slider)

    tabs = layout.add_element(Tabs())

    num_columns = 4

    tab1 = Div(num_columns=num_columns)

    for var_name, indices in transpose_list(
        [
            ('roll', slice(None)),
            ('pitch', slice(None)),
            ('thrust_scalar', slice(None)),
            ('thrust_scalar', slice(None)),
            #
            # ('masked_normal_distance_sunshade_detector_mm', slice(None)),
            ('masked_normal_distance_optics_detector_mm', slice(None)),
            # ('masked_distance_sunshade_optics_mm', slice(None)),
            ('masked_distance_optics_detector_mm', slice(None)),
            # #
            # ('normal_distance_sunshade_detector_mm', slice(None)),
            ('normal_distance_optics_detector_mm', slice(None)),
            # ('distance_sunshade_optics_mm', slice(None)),
            ('distance_optics_detector_mm', slice(None)),
            #
            ('orbit_state', 0),
            ('orbit_state', 1),
            ('relative_orbit_state', 0),
            ('relative_orbit_state', 1),
            # #
            # 'radius_km',
            # 'speed_km_s',
            # 'speed_km_s',
            # #
            # 'mass_flow_rate',
            # 'propellant_mass',
            # 'mass',
        ],
            num_columns):
        widget = Plot2D(
            xlabel='time',
            ylabel=var_name,
            xrange_mode='auto_scale_last_iter',
            yrange_mode='auto_scale_last_iter',
        )

        if 'mm' not in var_name:
            cubesat_names = [
                # 'sunshade_cubesat_group.',
                'optics_cubesat_group.',
                'detector_cubesat_group.',
            ]
        else:
            cubesat_names = [
                '',
            ]

        for cubesat_name in cubesat_names:
            get_x = generate_vector_func('times', generate_iopt_func('hist'))
            get_y = generate_vector_func_ind(
                '{}{}'.format(cubesat_name, var_name),
                generate_iopt_func('hist'), indices)
            widget.add_data_set(
                get_x=get_x,
                get_y=get_y,
                mode='lines+markers',
                marker_color='blue',
                line_color='blue',
            )
        tab1.add_element(widget)

    tabs.add_tab(tab1, 'Tab 1')

    # tab2 = Div(num_columns=1)

    # widget1 = Plot3D(
    #     xlabel='x',
    #     ylabel='y',
    #     zlabel='z',
    #     xrange_mode='auto_scale_last_iter',
    #     yrange_mode='auto_scale_last_iter',
    #     zrange_mode='auto_scale_last_iter',
    #     bg_color='black',
    # )

    # # for cubesat_name in [
    # #         'sunshade',
    # #         'optics',
    # #         'detector',
    # # ]:
    # # var1_name = 'position_km'
    # # var2_name = 'thrust_3xn'

    # widget1.add_data_set(
    #     type='scatter3d',
    #     get_x=generate_vector_func_ind('sunshade_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 0),
    #     get_y=generate_vector_func_ind('sunshade_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 1),
    #     get_z=generate_vector_func_ind('sunshade_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 2),
    #     mode='lines+markers',
    #     marker_color='red',
    #     line_color='red',
    #     marker_size=0.5,
    #     line_width=0.25,
    # )

    # widget1.add_data_set(
    #     type='scatter3d',
    #     get_x=generate_vector_func_ind('optics_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 0),
    #     get_y=generate_vector_func_ind('optics_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 1),
    #     get_z=generate_vector_func_ind('optics_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 2),
    #     mode='lines+markers',
    #     marker_color='white',
    #     line_color='white',
    #     marker_size=0.5,
    #     line_width=0.25,
    # )

    # widget1.add_data_set(
    #     type='scatter3d',
    #     get_x=generate_vector_func_ind('detector_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 0),
    #     get_y=generate_vector_func_ind('detector_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 1),
    #     get_z=generate_vector_func_ind('detector_cubesat_group.position_km',
    #                                    generate_iopt_func('hist'), 2),
    #     mode='lines+markers',
    #     marker_color='yellow',
    #     line_color='yellow',
    #     marker_size=0.5,
    #     line_width=0.25,
    # )

    # widget1.add_data_set(
    #     type='scatter3d',
    #     get_x=generate_vector_func_scaled_sum(
    #         'sunshade_cubesat_group.position_km',
    #         'detector_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 0),
    #     get_y=generate_vector_func_scaled_sum(
    #         'sunshade_cubesat_group.position_km',
    #         'detector_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 1),
    #     get_z=generate_vector_func_scaled_sum(
    #         'sunshade_cubesat_group.position_km',
    #         'sunshade_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 2),
    #     mode='markers',
    #     marker_color='red',
    #     marker_size=0.5,
    # )

    # widget1.add_data_set(
    #     type='scatter3d',
    #     get_x=generate_vector_func_scaled_sum(
    #         'optics_cubesat_group.position_km',
    #         'optics_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 0),
    #     get_y=generate_vector_func_scaled_sum(
    #         'optics_cubesat_group.position_km',
    #         'optics_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 1),
    #     get_z=generate_vector_func_scaled_sum(
    #         'optics_cubesat_group.position_km',
    #         'optics_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 2),
    #     mode='markers',
    #     marker_color='white',
    #     marker_size=0.5,
    # )

    # widget1.add_data_set(
    #     type='scatter3d',
    #     get_x=generate_vector_func_scaled_sum(
    #         'detector_cubesat_group.position_km',
    #         'detector_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 0),
    #     get_y=generate_vector_func_scaled_sum(
    #         'detector_cubesat_group.position_km',
    #         'detector_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 1),
    #     get_z=generate_vector_func_scaled_sum(
    #         'detector_cubesat_group.position_km',
    #         'detector_cubesat_group.thrust_3xn', 10000000,
    #         generate_iopt_func('hist'), 2),
    #     mode='markers',
    #     marker_color='yellow',
    #     marker_size=0.5,
    # )

    # widget1.add_data_set(type='mesh3d',
    #                      plot_func=generate_earth_stl_func('water'))
    # widget1.add_data_set(type='mesh3d',
    #                      plot_func=generate_earth_stl_func('land'))

    # tab2.add_element(widget1)

    # # for cubesat_name in [
    # #         'sunshade',
    # #         'optics',
    # #         'detector',
    # # ]:
    # widget2 = Plot3D(
    #     xlabel='x',
    #     ylabel='y',
    #     zlabel='z',
    #     xrange_mode='auto_scale_last_iter',
    #     yrange_mode='auto_scale_last_iter',
    #     zrange_mode='auto_scale_last_iter',
    #     bg_color='black',
    # )

    # widget2.add_data_set(type='mesh3d',
    #                         plot_func=generate_earth_stl_func('water'))
    # widget2.add_data_set(type='mesh3d',
    #                         plot_func=generate_earth_stl_func('land'))
    # widget2.add_data_set(
    #     type='scatter3d',
    #     get_x=generate_vector_func_ind(
    #         '{}_cubesat_group.position_km'.format(cubesat_name),
    #         generate_iopt_func('hist'), 0),
    #     get_y=generate_vector_func_ind(
    #         '{}_cubesat_group.position_km'.format(cubesat_name),
    #         generate_iopt_func('hist'), 1),
    #     get_z=generate_vector_func_ind(
    #         '{}_cubesat_group.position_km'.format(cubesat_name),
    #         generate_iopt_func('hist'), 2),
    #     mode='lines+markers',
    #     marker_color='red',
    #     line_color='red',
    #     marker_size=0.5,
    #     line_width=0.25,
    # )

    # widget2.add_data_set(
    #     type='scatter3d',
    #     get_x=generate_vector_func_scaled_sum(
    #         '{}_cubesat_group.position_km'.format(cubesat_name),
    #         '{}_cubesat_group.thrust_3xn'.format(cubesat_name), 10000000,
    #         generate_iopt_func('hist'), 0),
    #     get_y=generate_vector_func_scaled_sum(
    #         '{}_cubesat_group.position_km'.format(cubesat_name),
    #         '{}_cubesat_group.thrust_3xn'.format(cubesat_name), 10000000,
    #         generate_iopt_func('hist'), 1),
    #     get_z=generate_vector_func_scaled_sum(
    #         '{}_cubesat_group.position_km'.format(cubesat_name),
    #         '{}_cubesat_group.thrust_3xn'.format(cubesat_name), 10000000,
    #         generate_iopt_func('hist'), 2),
    #     mode='markers',
    #     marker_color='yellow',
    #     marker_size=0.5,
    # )

    # tab2.add_element(widget2)

    # tabs.add_tab(tab2, 'Tab 2')

    return layout
