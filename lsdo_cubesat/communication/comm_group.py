import numpy as np
from openmdao.api import ExecComp, IndepVarComp, Problem
from omtools.api import Group
import omtools.api as ot

# from lsdo_cubesat.api import Cubesat, Swarm
from lsdo_cubesat.attitude.rot_mtx_b_i_comp import RotMtxBIComp
from lsdo_cubesat.communication.Antenna_rot_mtx import AntennaRotationMtx
from lsdo_cubesat.communication.Antenna_rotation import AntRotationComp
from lsdo_cubesat.communication.Comm_Bitrate import BitRateComp
from lsdo_cubesat.communication.Comm_distance import \
    StationSatelliteDistanceComp
from lsdo_cubesat.communication.Comm_LOS import CommLOSComp
from lsdo_cubesat.communication.Comm_vector_antenna import AntennaBodyComp
from lsdo_cubesat.communication.Comm_VectorBody import VectorBodyComp
from lsdo_cubesat.communication.Data_download_rk4_comp import DataDownloadComp
from lsdo_cubesat.communication.Earth_spin_comp import EarthSpinComp
from lsdo_cubesat.communication.Earthspin_rot_mtx import EarthspinRotationMtx
from lsdo_cubesat.communication.GSposition_ECEF_comp import GS_ECEF_Comp
from lsdo_cubesat.communication.GSposition_ECI_comp import GS_ECI_Comp
# from lsdo_cubesat.communication.rot_mtx_ECI_EF_comp import RotMtxECIEFComp
from lsdo_cubesat.communication.Vec_satellite_GS_ECI import Comm_VectorECI
from lsdo_cubesat.ground_station_group import GSGroup
from lsdo_cubesat.options.ground_station import GroundStation
from lsdo_cubesat.utils.api import (ArrayExpansionComp, BsplineComp,
                                    ElementwiseMaxComp, LinearCombinationComp,
                                    PowerCombinationComp, get_bspline_mtx)


class CommGroup(Group):
    """
    Communication Discipline. Use one CommGroup per groundstation, per
    spacecraft.

    Options
    -------
    num_times : int
        Number of time steps over which to integrate dynamics
    step_size : float
        Constant time step size to use for integration
    num_cp : int
        Dimension of design variables/number of control points for
        BSpline components.
    ground_station : GroundStation
        GroundStation OptionsDictionary containing name, latitude,
        longitude, and altitude coordinates.
    mtx : array
        Matrix that translates control points (num_cp) to actual points
        (num_times)

    Parameters
    ----------
    orbit_state_km : shape=7
        Time history of orbit position (km) and velocity (km/s)
    rot_mtx_i_b_3x3n : shape=(3,3,n)
        Time history of rotation matrices from ECI to body frame

    Returns
    -------
    Download_rate
    P_comm
    """
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('ground_station', types=GroundStation)
        self.options.declare('mtx')

    def setup(self):

        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        mtx = self.options['mtx']
        ground_station = self.options['ground_station']
        name = ground_station['name']

        comp = IndepVarComp()
        comp.add_output('antenna_angle', val=1.6, units='rad')
        # comp.add_design_var('antenna_angle', lower=0., upper=10000)
        comp.add_output('P_comm_cp', val=13.0 * np.ones(num_cp), units='W')
        comp.add_design_var('P_comm_cp', lower=0., upper=20.0)
        comp.add_output('gain', val=16.0 * np.ones(num_times))
        comp.add_output('initial_data', val=0.0)

        self.add_subsystem('inputs_comp', comp, promotes=['*'])

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

        lat = ground_station['lat']
        lon = ground_station['lon']
        alt = ground_station['alt']

        r_e2g_E = np.zeros((3, num_times))
        Re = 6378.137
        r_GS = (Re + alt)
        d2r = np.pi / 180.

        cos_lat = np.cos(d2r * lat)

        r_e2g_E[0, :] = r_GS * cos_lat * np.cos(d2r * lon)
        r_e2g_E[1, :] = r_GS * cos_lat * np.sin(d2r * lon)
        r_e2g_E[2, :] = r_GS * np.sin(d2r * lat)

        r_e2g_I = self.create_indep_var(
            'r_e2g_I',
            val=np.einsum('ijn,jn->in', Rot_ECI_EF, r_e2g_E),
        )

        orbit_state_km = self.declare_input(
            'orbit_state_km',
            shape=(6, num_times),
        )
        r_b2g_I = self.register_output(
            'r_b2g_I',
            -(orbit_state_km[:3, :] - r_e2g_I),
        )
        comp = CommLOSComp(num_times=num_times)
        self.add_subsystem('CommLOS', comp, promotes=['*'])

        rot_mtx_i_b_3x3xn = self.declare_input(
            'rot_mtx_i_b_3x3xn',
            shape=(3, 3, num_times),
        )
        r_b2g_B = self.register_output(
            'r_b2g_B',
            ot.einsum(
                rot_mtx_i_b_3x3xn,
                r_b2g_I,
                subscripts='ijn,jn->in',
            ),
        )
        with self.create_group('antenna') as ant:
            rt2 = np.sqrt(2)
            antenna_angle = ant.declare_input('antenna_angle')
            antenna_orientation = ant.create_output(
                'antenna_orientation',
                shape=(4, num_times),
                val=0,
            )
            antenna_orientation[0, :] = ot.cos(
                ot.expand(antenna_angle, (1, num_times)) / 2) / rt2
            antenna_orientation[1, :] = ot.sin(
                ot.expand(antenna_angle, (1, num_times)) / 2) / rt2
            antenna_orientation[2, :] = -ot.sin(
                ot.expand(
                    antenna_angle,
                    (1, num_times),
                ) / 2, ) / rt2

        comp = AntennaRotationMtx(num_times=num_times)
        self.add_subsystem('antenna_rotation_mtx', comp, promotes=['*'])

        Rot_AB = self.declare_input('Rot_AB', shape=(3, 3, num_times))
        r_b2g_A = self.register_output(
            'r_b2g_A', ot.einsum(
                Rot_AB,
                r_b2g_B,
                subscripts='ijn,jn->in',
            ))
        distance_to_groundstation = self.register_output(
            'distance_to_groundstation',
            ot.pnorm(r_b2g_A, axis=0),
        )

        self.add_subsystem(
            'P_comm_comp',
            BsplineComp(
                num_pt=num_times,
                num_cp=num_cp,
                jac=mtx,
                in_name='P_comm_cp',
                out_name='P_comm',
            ),
            promotes=['*'],
        )

        self.add_subsystem(
            'Download_rate',
            BitRateComp(num_times=num_times),
            promotes=['*'],
        )


if __name__ == "__main__":
    from openmdao.api import Problem
    from openmdao.api import n2
    from lsdo_cubesat.utils.api import get_bspline_mtx
    num_times = 30
    num_cp = 10
    prob = Problem()
    prob.model = CommGroup(
        num_times=num_times,
        num_cp=num_cp,
        step_size=0.1,
        mtx=get_bspline_mtx(num_cp, num_times, order=4),
        ground_station=GroundStation(
            name='UCSD',
            lon=-117.1611,
            lat=32.7157,
            alt=0.4849,
        ),
    )
    prob.setup()
    # prob.run_model()
    n2(prob)
