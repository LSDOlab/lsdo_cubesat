from openmdao.api import Group, ExecComp

from lsdo_utils.api import get_bspline_mtx

from lsdo_cubesat.cubesat_group import CubesatGroup
from lsdo_cubesat.alignment.alignment_group import AlignmentGroup
from lsdo_cubesat.orbit.reference_orbit_group import ReferenceOrbitGroup


class SwarmGroup(Group):

    def initialize(self):
        self.options.declare('swarm')

    def setup(self):
        swarm = self.options['swarm']

        num_times = swarm['num_times']
        num_cp = swarm['num_cp']
        step_size = swarm['step_size']

        mtx = get_bspline_mtx(num_cp, num_times, order=4)

        group = ReferenceOrbitGroup(
            num_times=num_times,
            num_cp=num_cp,
            step_size=step_size,
            cubesat=swarm.children[0],
        )
        self.add_subsystem('reference_orbit_group', group, promotes=['*'])

        for cubesat in swarm.children:
            name = cubesat['name']

            group = CubesatGroup(
                num_times=num_times,
                num_cp=num_cp,
                step_size=step_size,
                cubesat=cubesat,
                mtx=mtx,
            )
            self.add_subsystem('{}_cubesat_group'.format(name), group)

        group = AlignmentGroup(
            swarm=swarm,
            mtx=mtx,
        )
        self.add_subsystem('alignment_group'.format(name),
                           group,
                           promotes=['*'])

        comp = ExecComp(
            'total_propellant_used' +
            '=sunshade_cubesat_group_total_propellant_used' +
            '+optics_cubesat_group_total_propellant_used' +
            '+detector_cubesat_group_total_propellant_used'
            # '+5.e-14*ks_masked_distance_sunshade_optics_km' +
            # '+5.e-14 *ks_masked_distance_optics_detector_km'
        )
        self.add_subsystem('total_propellant_used_comp', comp, promotes=['*'])

        for cubesat in swarm.children:
            name = cubesat['name']

            self.connect(
                '{}_cubesat_group.position_km'.format(name),
                '{}_cubesat_group_position_km'.format(name),
            )

            self.connect(
                '{}_cubesat_group.total_propellant_used'.format(name),
                '{}_cubesat_group_total_propellant_used'.format(name),
            )

            for var_name in [
                'radius',
                'reference_orbit_state',
            ]:
                self.connect(
                    var_name,
                    '{}_cubesat_group.{}'.format(name, var_name),
                )
