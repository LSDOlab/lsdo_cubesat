"""
RK4 component for Data Download
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.rk4_comp import RK4Comp


class DataDownloadComp(RK4Comp):
    def setup(self):
        opts = self.options
        n = opts['num_times']
        h = opts['step_size']
        
        # Inputs
        self.add_input('Download_rate',
                        np.zeros((1,n)),
                        desc='Data download rate over time')
        
        # Initial State
        self.add_input('Initial_Data',
                        np.zeros((1,)),
                        desc='Initial download data state')
        
        #States
        self.add_output('Data',
                        np.zeros((1,n)),
                        desc='Download data state over time')
        
        self.options['state_var'] = 'Data'
        self.options['init_state_var'] = 'Initial_Data'
        self.options['external_vars'] = ['Download_rate']
        
        self.dfdy = np.array([[0.]])
        self.dfdx = np.array([[1.]])
        
    def f_dot(self, external, state):
        return external[0]

    def df_dy(self, external, state):
        return self.dfdy

    def df_dx(self, external, state):
        return self.dfdx
        

if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp

    group = Group()

    comp = IndepVarComp()
    n = 2
    h = 6000.

    dd_dt = np.random.rand(1, n)
    Data0 = np.random.rand(1)
    comp.add_output('num_times', val=n)
    comp.add_output('Download_rate', val=dd_dt)
    comp.add_output('Initial_Data', val=Data0)

    group.add_subsystem('Inputcomp', comp, promotes=['*'])

    group.add_subsystem('Statecomp_Implicit',
                        DataDownloadComp(num_times=n, step_size=h),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
