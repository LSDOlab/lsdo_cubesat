import numpy as np

from openmdao.api import Group, IndepVarComp, NonlinearBlockGS, LinearBlockGS

from lsdo_utils.api import ArrayReorderComp, LinearCombinationComp, PowerCombinationComp

from lsdo_cubesat.utils.decompose_vector_group import DecomposeVectorGroup
from lsdo_cubesat.utils.mtx_vec_comp import MtxVecComp
from lsdo_cubesat.utils.ks_comp import KSComp
from lsdo_cubesat.orbit.initial_orbit_comp import InitialOrbitComp
from lsdo_cubesat.orbit.reference_orbit_rk4_comp import ReferenceOrbitRK4Comp
from lsdo_cubesat.orbit.orbit_state_decomposition_comp import OrbitStateDecompositionComp
from lsdo_cubesat.orbit.rot_mtx_t_i_comp import RotMtxTIComp


class ReferenceOrbitGroup(Group):

    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('num_cp', types=int)
        self.options.declare('step_size', types=float)
        self.options.declare('cubesat')

    def setup(self):
        num_times = self.options['num_times']
        num_cp = self.options['num_cp']
        step_size = self.options['step_size']
        cubesat = self.options['cubesat']

        shape = (3, num_times)

        drag_unit_vec = np.outer(
            np.array([0., 0., 1.]),
            np.ones(num_times),
        )

        comp = IndepVarComp()
        # comp.add_output('force_3xn', val=0., shape=shape)
        # comp.add_output('dry_mass', val=cubesat['dry_mass'], shape=num_times)
        comp.add_output('radius_earth_km',
                        val=cubesat['radius_earth_km'],
                        shape=num_times)
        for var_name in [
                'perigee_altitude',
                'apogee_altitude',
                'RAAN',
                'inclination',
                'argument_of_periapsis',
                'true_anomaly',
        ]:
            comp.add_output(var_name, val=cubesat[var_name])
        self.add_subsystem('input_comp', comp, promotes=['*'])

        # comp = LinearCombinationComp(
        #     shape=(num_times,),
        #     out_name='mass',
        #     coeffs_dict=dict(dry_mass=1., propellant_mass=1.),
        # )
        # self.add_subsystem('mass_comp', comp, promotes=['*'])

        comp = InitialOrbitComp()
        self.add_subsystem('initial_orbit_comp', comp, promotes=['*'])

        comp = ReferenceOrbitRK4Comp(
            num_times=num_times,
            step_size=step_size,
        )
        self.add_subsystem('orbit_rk4_comp', comp, promotes=['*'])

        comp = OrbitStateDecompositionComp(
            num_times=num_times, 
            position_name='position_km', 
            velocity_name='velocity_km_s',
            orbit_state_name='reference_orbit_state_km',
        )
        self.add_subsystem('orbit_state_decomposition_comp',
                           comp,
                           promotes=['*'])

        comp = LinearCombinationComp(
            shape=(6, num_times),
            out_name='reference_orbit_state',
            coeffs_dict=dict(reference_orbit_state_km=1.e3),
        )
        self.add_subsystem('position_comp', comp, promotes=['*'])

        group = DecomposeVectorGroup(
            num_times=num_times,
            vec_name='position_km',
            norm_name='radius_km',
            unit_vec_name='position_unit_vec',
        )
        self.add_subsystem('position_decomposition_group',
                           group,
                           promotes=['*'])

        comp = LinearCombinationComp(
            shape=(num_times,),
            out_name='radius',
            coeffs_dict=dict(radius_km=1.e3),
        )
        self.add_subsystem('radius_comp', comp, promotes=['*'])