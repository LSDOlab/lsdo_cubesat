"""
RK4 component for propellant state
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.rk4_comp import RK4Comp


class PropellantMassRK4Comp(RK4Comp):
    def setup(self):
        opts = self.options
        n = opts['num_times']
        h = opts['step_size']

        # Inputs
        self.add_input('mass_flow_rate',
                       np.zeros((1, n)),
                       desc='Propellant mass flow rate over time')

        # Initial State
        self.add_input('initial_propellant_mass',
                       1000 * np.ones((1, )),
                       desc='Initial propellant mass state')

        # States
        self.add_output('propellant_mass',
                        np.zeros((1, n)),
                        desc='Propellant mass state over time')

        self.options['state_var'] = 'propellant_mass'
        self.options['init_state_var'] = 'initial_propellant_mass'
        self.options['external_vars'] = ['mass_flow_rate']

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

    dm_dt = np.random.rand(1, n)
    Mass0 = np.random.rand(1)
    comp.add_output('num_times', val=n)
    comp.add_output('mass_flow_rate', val=dm_dt)
    comp.add_output('initial_propellant_mass', val=Mass0)

    group.add_subsystem('Inputcomp', comp, promotes=['*'])

    group.add_subsystem('Statecomp_Implicit',
                        PropellantMassComp(num_times=n, step_size=h),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
