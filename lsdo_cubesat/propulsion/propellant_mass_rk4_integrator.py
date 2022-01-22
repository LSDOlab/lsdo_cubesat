"""
RK4 component for propellant state
"""
import numpy as np

from lsdo_cubesat.operations.rk4_op import RK4


class PropellantMassRK4Integrator(RK4):

    def initialize(self):
        super().initialize()
        self.parameters.declare('num_times', types=int)

    def define(self):
        opts = self.parameters
        n = opts['num_times']

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

        self.parameters['state_var'] = 'propellant_mass'
        self.parameters['init_state_var'] = 'initial_propellant_mass'
        self.parameters['external_vars'] = ['mass_flow_rate']

        self.dfdy = np.array([[0.]])
        self.dfdx = np.array([[1.]])

    def f_dot(self, external, state):
        return external[0]

    def df_dy(self, external, state):
        return self.dfdy

    def df_dx(self, external, state):
        return self.dfdx


if __name__ == '__main__':

    from csdl import Model, custom
    from csdl_om import Simulator

    group = Model()

    num_times = 40
    step_size = 95 * 60 / (num_times - 1)

    dm_dt = np.random.rand(1, num_times)
    Mass0 = np.random.rand(1)
    mass_flow_rate = group.declare_variable('mass_flow_rate', val=dm_dt)
    initial_propellant_mass = group.declare_variable('initial_propellant_mass',
                                                     val=Mass0)

    propellant_mass = custom(
        mass_flow_rate,
        initial_propellant_mass,
        op=PropellantMassRK4Integrator(num_times=num_times,
                                       step_size=step_size),
    )
    group.register_output('propellant_mass', propellant_mass)

    sim = Simulator(group)
    sim.check_partials(compact_print=True)
