import numpy as np

from openmdao.api import ExplicitComponent


class RotMtxToRollPitchYaw(ExplicitComponent):
    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']
        self.add_input('rot_mtx_i_b_3x3xn', shape=(3, 3, num_times))
        self.add_output('roll', shape=(num_times))
        self.add_output('pitch', shape=(num_times))
        self.add_output('yaw', shape=(num_times))

        self.declare_partials('roll', 'rot_mtx_i_b_3x3xn', method='cs')
        self.declare_partials('pitch', 'rot_mtx_i_b_3x3xn', method='cs')
        self.declare_partials('yaw', 'rot_mtx_i_b_3x3xn', method='cs')

    def compute(self, inputs, outputs):
        R = inputs['rot_mtx_i_b_3x3xn_comp']
        outputs['roll'] = np.arctan2(np.array(R[1, 2, :], dtype=float),
                                     np.array(R[2, 2, :], dtype=float))
        outputs['pitch'] = -np.arcsin(R[0, 2, :])
        outputs['yaw'] = np.arctan2(np.array(R[0, 1, :], dtype=float),
                                    np.array(R[0, 0, :], dtype=float))


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp
    from lsdo_cubesat.attitude.inertia_ratios_comp import InertiaRatiosComp
    import matplotlib.pyplot as plt

    np.random.seed(0)
    num_times = 50

    class TestGroup(Group):
        def setup(self):
            comp = IndepVarComp()
            comp.add_output('rot_mtx_i_b_3x3xn_comp',
                            val=np.random.rand(9 * num_times).reshape(
                                (3, 3, num_times)))
            self.add_subsystem('inputs_comp', comp, promotes=['*'])
            self.add_subsystem('comp',
                               RotMtxToRollPitchYaw(num_times=num_times),
                               promotes=['*'])

    prob = Problem()
    prob.model = TestGroup()
    prob.setup(check=False)
    prob.run_model()
    prob.check_partials(compact_print=True)
