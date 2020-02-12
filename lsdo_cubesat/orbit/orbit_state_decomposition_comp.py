import numpy as np

from openmdao.api import ExplicitComponent


# 'position_km'
# 'velocity_km_s'

class OrbitStateDecompositionComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('position_name', types=str)
        self.options.declare('velocity_name', types=str)
        self.options.declare('orbit_state_name', types=str)

    def setup(self):
        num_times = self.options['num_times']
        position_name = self.options['position_name']
        velocity_name = self.options['velocity_name']
        orbit_state_name = self.options['orbit_state_name']

        self.add_input(orbit_state_name, shape=(6, num_times))
        self.add_output(position_name, shape=(3, num_times))
        self.add_output(velocity_name, shape=(3, num_times))

        orbit_state_indices = np.arange(6 * num_times).reshape((6, num_times))
        arange_3 = np.arange(3 * num_times)

        rows = arange_3
        cols = orbit_state_indices[:3, :].flatten()
        self.declare_partials(position_name, orbit_state_name, val=1., rows=rows, cols=cols)

        rows = arange_3
        cols = orbit_state_indices[3:, :].flatten()
        self.declare_partials(velocity_name, orbit_state_name, val=1., rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        position_name = self.options['position_name']
        velocity_name = self.options['velocity_name']
        orbit_state_name = self.options['orbit_state_name']

        outputs[position_name] = inputs[orbit_state_name][:3, :]
        outputs[velocity_name] = inputs[orbit_state_name][3:, :]


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output(orbit_state_name, np.random.rand(6, num_times))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = OrbitStateDecompositionComp(num_times=num_times)
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials()
