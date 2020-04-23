import numpy as np

from lsdo_viz.api import BaseViz, Frame
from lsdo_viz.api import write_stl_triangles, write_tecplot_dat_curve, write_stl_structured_list
#from lsdo_viz.api import get_sphere_triangulation, get_earth_triangulation, write_paraview

#from lsdo_viz.api import get_sphere_triangulation, get_earth_triangulation, write_paraview


import seaborn as sns


# sns.set()


earth_radius = 6371.
cubesat_names = ['sunshade', 'optics', 'detector']

class Viz(BaseViz):

    def setup(self):
        self.frame_name_format = 'output_{}'

        self.add_frame(Frame(
            height_in=6., width_in=9.,
            nrows=1, ncols=1,
            wspace=0.4, hspace=0.4,
        ), 1)

        self.sphere_triangulation = get_sphere_triangulation(earth_radius, 200, 200)
        self.earth_triangulation = get_earth_triangulation(earth_radius)

    def plot(self, data_dict_list, ind, video=False):
        import matplotlib.image as mpimg

        position_km_dict = dict()
        reference_orbit_dict = dict()
        relative_orbit_dict = dict()

        for cubesat_name in cubesat_names:
            reference_orbit = data_dict_list[ind]['reference_orbit_state']
            relative_orbit = data_dict_list[ind]['{}_cubesat_group.relative_orbit_state'.format(cubesat_name)]
            thrust_3xn = data_dict_list[ind]['{}_cubesat_group.thrust_3xn'.format(cubesat_name)]

        write_stl_triangles('viz/earth_sphere', self.sphere_triangulation)
        write_stl_triangles('viz/earth_triangulation', self.earth_triangulation)

        with self.get_frame(1)[0, 0] as ax:
            ax.axis('off')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            sns.axes_style({
                'axes.grid': False,
                'axes.spines.left': False,
                'axes.spines.bottom': False,
                'axes.spines.right': False,
                'axes.spines.top': False,
            })

            thrust_3xn = data_dict_list[ind]['{}_cubesat_group.thrust_3xn'.format(cubesat_name)]
            num_times = thrust_3xn.shape[1]

            itimes = range(150, num_times, 150)
            # itimes = [1500]
            for itime in itimes:
                print('ind, itime', ind, itime)
                self.get_frame(1).clear_all_axes()

                for cubesat_name in cubesat_names:
                    reference_orbit = data_dict_list[ind]['reference_orbit_state']
                    relative_orbit = data_dict_list[ind]['{}_cubesat_group.relative_orbit_state'.format(cubesat_name)]
                    thrust_3xn = data_dict_list[ind]['{}_cubesat_group.thrust_3xn'.format(cubesat_name)]

                    reference_orbit_km = reference_orbit.T / 1.e3
                    relative_orbit_km = relative_orbit.T / 1.e3

                    # print('ref', reference_orbit_km)
                    # print('rel', relative_orbit_km)
                    # print('rel', reference_orbit_km + relative_orbit_km)
                    # print()
                    # import matplotlib.pyplot as plt
                    # plt.plot(relative_orbit_km[:, 0])
                    # plt.plot(relative_orbit_km[:, 1])
                    # plt.plot(relative_orbit_km[:, 2])
                    # plt.savefig('t1.pdf')
                    # exit()

                    position_km = reference_orbit_km + 1.e5 * relative_orbit_km
                    position_km = position_km[:, :3]

                    num_times = position_km.shape[0]
                    data = np.zeros((num_times, 2, 3))
                    data[:, 0, :] = position_km
                    data[:, 1, :] = position_km + thrust_3xn.T * 1.e5
                
                    write_tecplot_dat_curve('viz/{}_position'.format(cubesat_name), position_km[:itime, :])
                    write_tecplot_dat_curve('viz/{}_position_ends'.format(cubesat_name), position_km[(0, itime - 1), :])
                    write_stl_structured_list('viz/{}_thrust'.format(cubesat_name), [data[:itime, :]])

                write_paraview('viz/paraview_script', 'viz/screen')

                img = mpimg.imread('viz/screen.png')

                ax.imshow(img)

                self.get_frame(1).write()
