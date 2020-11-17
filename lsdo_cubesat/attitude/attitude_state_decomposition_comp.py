import numpy as np

from openmdao.api import ExplicitComponent

# 'position_km'
# 'velocity_km_s'


class AttitudeStateDecompositionComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('angular_velocity_name', types=str)
        self.options.declare('quaternion_name', types=str)
        self.options.declare('angular_velocity_orientation', types=str)

    def setup(self):
        num_times = self.options['num_times']
        angular_velocity_name = self.options['angular_velocity_name']
        quaternion_name = self.options['quaternion_name']
        attitude_state_name = self.options[
            'angular_velocity_orientation']

        self.add_input(attitude_state_name, shape=(7, num_times))
        self.add_output(angular_velocity_name, shape=(3, num_times))
        self.add_output(quaternion_name, shape=(4, num_times))

        attitude_state_indices = np.arange(7 * num_times).reshape(
            (7, num_times))

        rows = np.arange(3 * num_times)
        cols = attitude_state_indices[:3, :].flatten()
        self.declare_partials(
            angular_velocity_name,
            attitude_state_name,
            val=1.,
            rows=rows,
            cols=cols)

        rows = np.arange(4 * num_times)
        cols = attitude_state_indices[3:, :].flatten()
        self.declare_partials(
            quaternion_name,
            attitude_state_name,
            val=1.,
            rows=rows,
            cols=cols)

    def compute(self, inputs, outputs):
        angular_velocity_name = self.options['angular_velocity_name']
        quaternion_name = self.options['quaternion_name']
        attitude_state_name = self.options[
            'angular_velocity_orientation']

        outputs[angular_velocity_name] = inputs[
            attitude_state_name][:3, :]
        outputs[quaternion_name] = inputs[attitude_state_name][3:, :]


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output(
        'angular_velocity_orientation', np.random.rand(7, num_times))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = AttitudeStateDecompositionComp(
        num_times=num_times,
        angular_velocity_orientation='angular_velocity_orientation',
        angular_velocity_name='angular_velocity',
        quaternion_name='quaternions')
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
    # prob.check_partials()
