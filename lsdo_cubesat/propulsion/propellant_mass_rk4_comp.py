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
    """
    Integrate propellant mass flow rate over time using Runge-Kutta 4
    method

    Options
    ----------
    num_times : int
        Number of time steps over which to integrate dynamics
    step_size : float
        Constant time step size to use for integration

    Parameters
    ----------
    mass_flow_rate : shape=(1, num_times)
        Time history of mass flow rate
    initial_propellant_mass : shape=(1,)
        Initial propellant mass

    Returns
    -------
    propellant_mass : shape=(1, num_times)
        Time history of propellant mass

    """
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
