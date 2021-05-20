import numpy as np
from omtools.api import Group

class StationSatelliteDistanceGroup(Group):
    """
    Calculates distance from ground station to satellite.
    """

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.register_output('GSdist', np.zeros(num_times))
#        units='km',
#                        desc='Distance from ground station to satellite over time'

        self.declare_input('r_b2g_A', np.zeros((3, num_times)))
#        units='km',
#                      desc='Position vector from satellite to ground station '
#                            'in antenna angle frame over time'



if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 4
    comp.add_output('r_b2g_A', val=np.random.random((3, num_times)),units='km')

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('distance',
                        StationSatelliteDistanceGroup(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
