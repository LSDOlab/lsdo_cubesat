import numpy as np

from openmdao.api import ExplicitComponent


class InertiaRatiosComp(ExplicitComponent):
    def setup(self):

        self.add_input('mass_moment_inertia_b_frame_km_m2', shape=(3))
        self.add_output('moment_inertia_ratios', shape=(3))

        self.declare_partials('moment_inertia_ratios',
                              'mass_moment_inertia_b_frame_km_m2')

    def compute(self, inputs, outputs):
        I = inputs['mass_moment_inertia_b_frame_km_m2']
        K = outputs['moment_inertia_ratios']
        K[0] = (I[1] - I[2]) / I[0]
        K[1] = (I[2] - I[0]) / I[1]
        K[2] = (I[0] - I[1]) / I[2]

    def compute_partials(self, inputs, partials):
        I = inputs['mass_moment_inertia_b_frame_km_m2']
        dKdI = np.zeros((3, 3))

        dKdI[0, 0] = -(I[1] - I[2]) / I[0]**2
        dKdI[0, 1] = 1 / I[0]
        dKdI[0, 2] = -1 / I[0]
        dKdI[1, 0] = -1 / I[1]
        dKdI[1, 1] = -(I[2] - I[0]) / I[1]**2
        dKdI[1, 2] = 1 / I[1]
        dKdI[2, 0] = 1 / I[2]
        dKdI[2, 1] = -1 / I[2]
        dKdI[2, 2] = -(I[0] - I[1]) / I[2]**2

        partials['moment_inertia_ratios',
                 'mass_moment_inertia_b_frame_km_m2'] = dKdI


if __name__ == '__main__':

    from openmdao.api import Problem, Group, IndepVarComp

    np.random.seed(1)

    prob = Problem()
    inputs = IndepVarComp()
    inputs.add_output('mass_moment_inertia_b_frame_km_m2',
                      val=np.random.rand(3),
                      shape=(3))
    prob.model.add_subsystem('inputs', inputs, promotes=['*'])
    prob.model.add_subsystem('inertia_ratios',
                             InertiaRatiosComp(),
                             promotes=['*'])

    prob.setup()
    prob.run_model()
    print(prob['moment_inertia_ratios'])
    prob.check_partials(compact_print=True)
