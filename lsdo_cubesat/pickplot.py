import pickle
# import pyproj
# import cartopy.crs as ccrs
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# from lsdo_cubesat.utils.comps.arithmetic_comps.elementwise_max_comp import ElementwiseMaxComp
# from lsdo_cubesat.utils.comps.arithmetic_comps.elementwise_max_comp import ElementwiseMaxComp

# path = "/home/lsdo/Cubesat/lsdo_cubesat/1/_data/opt.00478.pkl"
path = "/home/lsdo/Cubesat/lsdo_cubesat/_data/opt.00583.pkl"
# path = '/home/lsdo/1_result/result_7.22/_data/opt.01521.pkl'
# path = '/home/lsdo/1_result/latest_result/data_file/opt.07971.pkl'
# path = '/home/lsdo/Cubesat/lsdo_cubesat/1314/_data_123/opt.03198.pkl'

with open(path, 'rb') as f:
    info = pickle.load(f)
# print(info)

X_reference = info['reference_orbit_state'][0, :]
Y_reference = info['reference_orbit_state'][1, :]
Z_reference = info['reference_orbit_state'][2, :]

# X_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][0, :]
# Y_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][1, :]
# Z_sunshade_relative = info['sunshade_cubesat_group.relative_orbit_state'][2, :]

X_detector_relative = info['detector_cubesat_group.relative_orbit_state'][0, :]
Y_detector_relative = info['detector_cubesat_group.relative_orbit_state'][1, :]
Z_detector_relative = info['detector_cubesat_group.relative_orbit_state'][2, :]

X_optics_relative = info['optics_cubesat_group.relative_orbit_state'][0, :]
Y_optics_relative = info['optics_cubesat_group.relative_orbit_state'][1, :]
Z_optics_relative = info['optics_cubesat_group.relative_orbit_state'][2, :]

time = np.arange((X_detector_relative.shape[0]))

x = np.linspace(-10, 10, 100)


def sigmoid(x):
    return 1 / (1 + np.exp(-100 * x))


reference_matrix = info["reference_orbit_state"][:3, :]
r_orbit = np.linalg.norm(reference_matrix, ord=1, axis=0)

optics_matrix = info["optics_cubesat_group.relative_orbit_state"][:3, :]
r_optics = np.linalg.norm(optics_matrix, ord=1, axis=0)

detector_matrix = info["detector_cubesat_group.relative_orbit_state"][:3, :]
r_detector = np.linalg.norm(detector_matrix, ord=1, axis=0)

# sunshade_matrix = info["sunshade_cubesat_group.relative_orbit_state"][:3, :]
# r_sunshade = np.linalg.norm(sunshade_matrix, ord=1, axis=0)

time = np.arange(r_orbit.shape[0])

theta = 2 * np.pi / (r_orbit.shape[0]) * time

# A = info["normal_distance_sunshade_detector_mm"]
B = info["normal_distance_optics_detector_mm"]
# C = info["distance_sunshade_optics_mm"]
D = info["distance_optics_detector_mm"]
E = info["detector_cubesat_group.roll"]
F = info["detector_cubesat_group.pitch"]
# G = info["detector_cubesat_group.Data"]
# Gr = info["detector_cubesat_group.Download_rate"]
# P = info["detector_cubesat_group.P_comm"]
# M = info["detector_cubesat_group.propellant_mass"]
# obj = info["obj"]
# print(G.shape)

mask_vec = info['mask_vec']

# sunshade_UCSD_Data = info[
#     'sunshade_cubesat_group.UCSD_comm_group.total_data_downloaded']
# sunshade_UIUC_Data = info[
#     'sunshade_cubesat_group.UIUC_comm_group.total_data_downloaded']
# sunshade_Georgia_Data = info[
#     'sunshade_cubesat_group.Georgia_comm_group.total_data_downloaded']
# sunshade_Montana_Data = info[
#     'sunshade_cubesat_group.Montana_comm_group.total_data_downloaded']

# masked_normal_distance_sunshade_detector = info[ 'masked_normal_distance_sunshade_detector_mm_sq_sum']
masked_normal_distance_optics_detector = info[
    'masked_normal_distance_optics_detector_mm_sq_sum']
# masked_distance_sunshade_optics = info[ 'masked_distance_sunshade_optics_mm_sq_sum']
masked_distance_optics_detector = info[
    'masked_distance_optics_detector_mm_sq_sum']
optics_relative = info['optics_cubesat_group.relative_orbit_state_sq_sum']
detector_relative = info['detector_cubesat_group.relative_orbit_state_sq_sum']
# sunshade_relative = info["sunshade_cubesat_group.relative_orbit_state_sq_sum"]
obj = info['obj']

# KS_data = info['detector_cubesat_group.KS_Data']
UCSD_data_rate = info['detector_cubesat_group.UCSD_comm_group.Download_rate']
GS_dist = info['detector_cubesat_group.UCSD_comm_group.GSdist']

# sunshade_UCSD_data = info['detector_cubesat_group.UCSD_comm_group.Data']
# sunshade_UIUC_data = info['detector_cubesat_group.UIUC_comm_group.Data']
# sunshade_Georgia_data = info['detector_cubesat_group.Georgia_comm_group.Data']
# sunshade_Montana_data = info['detector_cubesat_group.Montana_comm_group.Data']

# sunshade_UCSD_LOS = info['detector_cubesat_group.UCSD_comm_group.CommLOS']
# sunshade_UIUC_LOS = info['detector_cubesat_group.UIUC_comm_group.CommLOS']
# sunshade_Georgia_LOS = info['detector_cubesat_group.Georgia_comm_group.CommLOS']
# sunshade_Montana_LOS = info['detector_cubesat_group.Montana_comm_group.CommLOS']

# print(masked_normal_distance_optics_detector / 1500)
# print(masked_normal_distance_optics_detector / 1500)
# print(masked_distance_optics_detector / 1500)
# print(masked_distance_sunshade_optics / 1500)

# print(optics_relative / 1500)
# print(detector_relative / 1500)
# print(sunshade_relative / 1500)
time = 1501
num_times = np.arange(time)

# print(sunshade_UCSD_LOS.shape)
# KS_data = KS_data.reshape(-1, )
# print(num_times.shape)
# print(KS_data.shape)

# plt.plot(num_times, KS_data)
# plt.legend()
# plt.show()

position = info['detector_cubesat_group.position']
velocity = info['detector_cubesat_group.velocity']

position = np.linalg.norm(position, ord=1, axis=0)
velocity = np.linalg.norm(velocity, ord=1, axis=0)
# print(Gr)
# print(M.shape)

time = np.arange(1501)

print(obj)
# print(masked_normal_distance_sunshade_detector / 1500)
print(masked_normal_distance_optics_detector / 1500)
# print(masked_distance_sunshade_optics / 1500)
print(masked_distance_optics_detector / 1500)
# print(sunshade_relative / 1500)
print(optics_relative / 1500)
print(detector_relative / 1500)
print(info['total_data_downloaded'])
print(info['total_propellant_used'])
# print(np.min(info['observation_dot']))
# print(reference_matrix)

# if __name__ == '__main__':
#     import numpy as np

#     from openmdao.api import Problem, IndepVarComp, Group

#     sunshade_UCSD_data = info['sunshade_cubesat_group.UCSD_comm_group.Data']
#     sunshade_UIUC_data = info['sunshade_cubesat_group.UIUC_comm_group.Data']
#     sunshade_Georgia_data = info[
#         'sunshade_cubesat_group.Georgia_comm_group.Data']
#     sunshade_Montana_data = info[
#         'sunshade_cubesat_group.Montana_comm_group.Data']

#     group = Group()
#     comp = IndepVarComp()
#     num_times = 1501
#     shape = (1, num_times)
#     rho = 100.
#     comp.add_output('UCSD', val=sunshade_UCSD_data)
#     comp.add_output('UIUC', val=sunshade_UIUC_data)
#     comp.add_output('Georgia', val=sunshade_Georgia_data)
#     comp.add_output('Montana', val=sunshade_Montana_data)

#     group.add_subsystem('Inputcomp', comp, promotes=['*'])
#     group.add_subsystem('KS_Data',
#                         ElementwiseMaxComp(shape=shape,
#                                            in_names=[
#                                                'UCSD',
#                                                'UIUC',
#                                                'Georgia',
#                                                'Montana',
#                                            ],
#                                            out_name='KS_Data',
#                                            rho=rho),
#                         promotes=['*'])

#     prob = Problem()
#     prob.model = group
#     prob.setup(check=True)
#     prob.run_model()
#     prob.model.list_outputs()

#     prob.check_partials(compact_print=True)

# print(prob['UCSD_comm_group_Data'])

sns.set()

# plt.plot(time, prob['UCSD'].T, label='UCSD')
# plt.plot(time, prob['UIUC'].T, label='UIUC')
# plt.plot(time, prob['Georgia'].T, label='Georgia')
# plt.plot(time, prob['Montana'].T, label='Montana')
# plt.plot(time, sunshade_UCSD_data.T, label='UCSD')
# plt.plot(time, sunshade_UIUC_data.T, label='UIUC')
# plt.plot(time, sunshade_Georgia_data.T, label='Georgia')
# plt.plot(time, sunshade_Montana_data.T, label='Montana')
# plt.plot(time, prob['KS_Data'].T, label='KS')
# plt.legend()
# plt.show()
# # C = np.abs(sunshade_matrix - detector_matrix)
# # D = np.linalg.norm(C, ord=1, axis=0)
# plt.plot(time, KS_data.T)
# # plt.plot(time, np.abs(r_optics - r_sunshade), label='optics-sunshade')
# # plt.plot(time, np.abs(r_optics - r_detector), label='optics-detector')
# # plt.plot(time, np.abs(r_detector - r_sunshade), label='detector-sunshade')
# plt.plot(time, position)
# plt.plot(time, velocity)
# plt.plot(time, info['mask_vec'], label='mask_vec')
# plt.plot(time, info['observation_dot'], label='observation_dot')
# plt.plot(time, A, label="alignment_s_d")
plt.plot(time, B, label="alignment_o_d")
# plt.plot(time, C, label="seperation_s_o")
plt.plot(time, D, label="seperation_o_d")
# plt.plot(time, sunshade_UCSD_data.T)
# plt.plot(time, sunshade_UIUC_data.T)
# plt.plot(time, sunshade_Georgia_data.T)
# plt.plot(time, sunshade_Montana_data.T)

# plt.plot(time, sunshade_UCSD_LOS, label='UCSD')
# plt.plot(time, sunshade_UIUC_LOS, label='UIUC')
# plt.plot(time, sunshade_Georgia_LOS, label='Georgia')
# plt.plot(time, sunshade_Montana_LOS, label='Montana')

# # plt.plot(time, KS_data.T)
# # plt.plot(time, GS_dist)
# # plt.plot(time, sunshade_relative)
# # plt.plot(time, detector_relative)
# # plt.plot(time, optics_relative)

# # plt.plot(time, r_detector - r_optics, label='detector-optics')
# # plt.plot(time,
# #          np.abs((sunshade_matrix - detector_matrix)[2, :]),
# #          label='sunshade-detector')
# # plt.plot(time, np.abs(r_sunshade - r_optics), label='sunshade-optics')
# # plt.plot(time, r_orbit / 10000 + r_detector)
plt.legend()
plt.show()
# # ax = plt.subplot(121, projection='polar')
# # ax.plot(theta, r_orbit / 10000)
# # ax.plot(theta, r_optics + r_orbit / 10000)

# # # plt.plot(time, r_orbit / 10000)
# # # plt.plot(time, r_orbit / 10000)
# # plt.show()

# # X_detector_new = sigmoid(X_detector_relative)
# # Y_detector_new = sigmoid(Y_detector_relative)
# # Z_detector_new = sigmoid(Z_detector_relative)

# # X_sunshade_new = sigmoid(X_sunshade_relative)
# # Y_sunshade_new = sigmoid(Y_sunshade_relative)
# # Z_sunshade_new = sigmoid(Z_sunshade_relative)

# # plt.plot(x, z)
# # plt.xlabel("x")
# # plt.ylabel("Sigmoid(X)")
# # plt.show()

# # ax.plot(X_reference / 1000 + X_detector_relative,
# #         Y_reference / 1000 + Y_detector_relative,
# #         Z_reference / 1000 + Z_detector_relative)

# # plt.plot(Y_reference / 100000 + Y_detector_relative * 100,
# #          Z_reference / 100000 + Y_detector_relative * 100)

# # plt.plot(X, Y)
# # plt.ylim(ymin=0.1699)
# # plt.ylim(ymax=0.170025)
# # plt.xlim(xmin=0.0)
# # plt.xlabel("time")
# # plt.ylabel("reference_orbit_state")
# # plt.legend()
# # plt.grid()
# # plt.savefig("Data_downloaded.png", dpi=120)
# # plt.show()
