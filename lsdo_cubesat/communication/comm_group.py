import numpy as np

from csdl import Model
import csdl
from csdl.utils.get_bspline_mtx import get_bspline_mtx

from lsdo_cubesat.swarm.ground_station import Ground_station
from lsdo_cubesat.swarm.GS_net import GS_net
from lsdo_cubesat.ground_station_group import GSGroup
from lsdo_cubesat.attitude.rot_mtx_b_i_comp import RotMtxBIComp
from lsdo_cubesat.communication.Antenna_rot_mtx import AntennaRotationMtx
from lsdo_cubesat.communication.Antenna_rotation import AntRotationComp
from lsdo_cubesat.communication.Comm_distance import StationSatelliteDistanceComp
from lsdo_cubesat.communication.Comm_LOS import CommLOSComp
from lsdo_cubesat.communication.Comm_VectorBody import VectorBodyComp
from lsdo_cubesat.communication.GSposition_ECEF_comp import GS_ECEF_Comp
from lsdo_cubesat.communication.GSposition_ECI_comp import GS_ECI_Comp
# from lsdo_cubesat.communication.rot_mtx_ECI_EF_comp import RotMtxECIEFComp
from lsdo_cubesat.communication.Vec_satellite_GS_ECI import Comm_VectorECI
from lsdo_cubesat.communication.Comm_vector_antenna import AntennaBodyComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Earth_spin_comp import EarthSpinComp
from lsdo_cubesat.communication.Earthspin_rot_mtx import EarthspinRotationMtx
# from lsdo_cubesat.communication.Ground_comm import Groundcomm
from lsdo_cubesat.constants import RADII
from lsdo_cubesat.communication.Comm_LOS import CommLOSComp
from lsdo_cubesat.communication.bitrate import BitRate

radius_earth = RADII['Earth']


def LOS(self, x):
    b = np.argsort(x)
    x[b[:50]] = 1
    x[b[50:]] = 0
    return x


class CommGroup(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('num_cp', types=int)
        self.parameters.declare('step_size', types=float)

        # self.parameters.declare('cubesat')
        self.parameters.declare('ground_station')

    def define(self):

        num_times = self.parameters['num_times']
        num_cp = self.parameters['num_cp']
        step_size = self.parameters['step_size']
        ground_station = self.parameters['ground_station']

        # TODO: review values of inputs
        P_comm_cp = self.create_input('P_comm_cp',
                                      val=13.0 * np.ones(num_cp),
                                      units='W')
        self.add_design_variable('P_comm_cp', lower=0., upper=20.0)
        P_comm = csdl.matvec(get_bspline_mtx(num_cp, num_times), P_comm_cp)
        self.register_output('P_comm', P_comm)

        antAngle = self.create_input('antAngle', val=1.6, units='rad')
        # comp.add_design_var('antAngle', lower=0., upper=10000)
        gain = self.create_input('gain', val=16.0 * np.ones(num_times))

        q_E = np.zeros((4, num_times))
        t = step_size * np.arange(num_times)
        theta = t * 2 * np.pi / 3600.0 / 24.0
        q_E[0, :] = np.cos(theta)
        q_E[3, :] = -np.sin(theta)

        Rot_ECI_EF = np.zeros((3, 3, num_times))
        Rot_ECI_EF[0, 0, :] = 1 - 2 * q_E[2, :]**2 - 2 * q_E[3, :]**2
        Rot_ECI_EF[
            0, 1, :] = 2 * q_E[1, :] * q_E[2, :] - 2 * q_E[3, :] * q_E[0, :]
        Rot_ECI_EF[
            0, 2, :] = 2 * q_E[1, :] * q_E[3, :] + 2 * q_E[2, :] * q_E[0, :]
        Rot_ECI_EF[
            1, 0, :] = 2 * q_E[1, :] * q_E[2, :] + 2 * q_E[3, :] * q_E[0, :]
        Rot_ECI_EF[1, 1, :] = 1 - 2 * q_E[1, :]**2 - 2 * q_E[3, :]**2
        Rot_ECI_EF[
            1, 2, :] = 2 * q_E[2, :] * q_E[3, :] - 2 * q_E[1, :] * q_E[0, :]
        Rot_ECI_EF[
            2, 0, :] = 2 * q_E[1, :] * q_E[3, :] - 2 * q_E[2, :] * q_E[0, :]
        Rot_ECI_EF[
            2, 1, :] = 2 * q_E[2, :] * q_E[3, :] + 2 * q_E[1, :] * q_E[0, :]
        Rot_ECI_EF[2, 2, :] = 1 - 2 * q_E[1, :]**2 - 2 * q_E[2, :]**2

        # degrees
        lat = ground_station['lat'] * np.pi / 180.
        lon = ground_station['lon'] * np.pi / 180.
        alt = ground_station['alt'] * np.pi / 180.

        # position of the ground station in Earth-fixed frame
        r_e2g_E = np.zeros((3, num_times))
        radius_earth = 6378.137
        r_GS = (radius_earth + alt)

        r_e2g_E[0, :] = r_GS * np.cos(lat) * np.cos(lon)
        r_e2g_E[1, :] = r_GS * np.cos(lat) * np.sin(lon)
        r_e2g_E[2, :] = r_GS * np.sin(lat)

        r_e2g_I = self.create_input(
            'r_e2g_I',
            val=np.einsum('ijn,jn->in', Rot_ECI_EF, r_e2g_E),
        )

        orbit_state_km = self.declare_variable(
            'orbit_state_km',
            shape=(6, num_times),
        )

        # Position vector from satellite to ground station in
        # Earth-centered inertial frame over time
        r_b2g_I = self.register_output(
            'r_b2g_I',
            -(orbit_state_km[:3, :] - r_e2g_I),
        )

        CommLOS = csdl.custom(
            r_b2g_I,
            r_e2g_I,
            op=CommLOSComp(num_times=num_times),
        )
        self.register_output(
            'CommLOS',
            CommLOS,
        )

        rot_mtx_i_b_3x3xn = self.declare_variable(
            'rot_mtx_i_b_3x3xn',
            shape=(3, 3, num_times),
        )
        r_b2g_B = self.register_output(
            'r_b2g_B',
            csdl.einsum(
                rot_mtx_i_b_3x3xn,
                r_b2g_I,
                subscripts='ijn,jn->in',
            ),
        )
        with self.create_submodel('antenna') as ant:
            rt2 = np.sqrt(2)
            # TODO: nothing computes antenna angle, even in Aobo's
            # original code
            antenna_angle = ant.declare_variable('antenna_angle', val=0.0)
            q_A = ant.create_output(
                'q_A',
                shape=(4, num_times),
                val=0,
            )
            q_A[0, :] = csdl.cos(
                csdl.expand(antenna_angle, (1, num_times)) / 2) / rt2
            q_A[1, :] = csdl.sin(
                csdl.expand(antenna_angle, (1, num_times)) / 2) / rt2
            q_A[2, :] = -csdl.sin(
                csdl.expand(
                    antenna_angle,
                    (1, num_times),
                ) / 2, ) / rt2

        Rot_AB = csdl.custom(q_A, op=AntennaRotationMtx(num_times=num_times))

        self.register_output('Rot_AB', Rot_AB)  #, shape=(3, 3, num_times))
        r_b2g_A = csdl.einsum(
            Rot_AB,
            r_b2g_B,
            subscripts='ijn,jn->in',
        )
        self.register_output('r_b2g_A', r_b2g_A)

        distance_to_groundstation = self.register_output(
            'GSdist',
            csdl.pnorm(r_b2g_A, axis=0),
        )

        Download_rate = csdl.custom(
            P_comm,
            gain,
            distance_to_groundstation,
            CommLOS,
            op=BitRate(num_times=num_times),
        )
        self.register_output(
            'Download_rate',
            Download_rate,
        )
