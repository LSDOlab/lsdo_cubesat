import numpy as np
from openmdao.api import ExplicitComponent

class StationSatelliteDistanceComp(ExplicitComponent):
    """
    Calculates distance from ground station to satellite.
    """

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('r_b2g_A', np.zeros((3, num_times)), units='km',
                       desc='Position vector from satellite to ground station '
                            'in antenna angle frame over time')

        self.add_output('GSdist', np.zeros(num_times), units='km',
                        desc='Distance from ground station to satellite over time')

        self.declare_partials('GSdist','r_b2g_A')

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']

        r_b2g_A = inputs['r_b2g_A']

        outputs['GSdist'] = np.linalg.norm(r_b2g_A,axis=0)


    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        r_b2g_A = inputs['r_b2g_A']

        dGS_drot = np.zeros((num_times,3,num_times))
    
        norm = np.linalg.norm(r_b2g_A, axis=0)
        C = r_b2g_A/norm

        for i in range(0,num_times):
            dGS_drot[i,:,i] = C[:,i]

        partials['GSdist','r_b2g_A'] = dGS_drot.flatten()



if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 4
    comp.add_output('r_b2g_A', val=np.random.random((3, num_times)),units='km')

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('distance',
                        StationSatelliteDistanceComp(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
