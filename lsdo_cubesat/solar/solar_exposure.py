from csdl import CustomExplicitOperation
from smt.surrogate_models import RMTB

import numpy as np


class SolarExposure(CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']
        self.add_input('sun_component', shape=(1, num_times))
        self.add_output('percent_exposed_area', shape=(1, num_times))
        r = np.arange(num_times)
        self.declare_derivatives('percent_exposed_area',
                                 'sun_component',
                                 rows=r,
                                 cols=r)
        self.sm = RMTB(
            xlimits=np.array([[-1., 1.]]),
            order=4,
            num_ctrl_pts=20,
            energy_weight=1e-3,
            regularization_weight=1e-7,
            print_global=False,
        )
        xt = np.linspace(-1, 1, 10000)
        yt = np.where(xt < 0, 0, xt)
        self.sm.set_training_values(xt, yt)
        self.sm.train()

    def compute(self, inputs, outputs):
        c = inputs['sun_component'].reshape(-1, 1)
        outputs['percent_exposed_area'] = self.sm.predict_values(c)

    def compute_derivatives(self, inputs, derivatives):
        c = inputs['sun_component'].reshape(-1, 1)
        derivatives['percent_exposed_area',
                    'sun_component'] = self.sm.predict_derivatives(
                        c, 0).flatten()


if __name__ == "__main__":
    from csdl import Model
    import csdl
    from csdl_om import Simulator

    class M(Model):
        def define(self):
            num_times = 5
            sun_component = self.declare_variable(
                'sun_component',
                shape=(1, num_times),
                val=np.random.rand(num_times).reshape((1, num_times)) - 0.5,
            )
            percent_exposed_area = csdl.custom(
                sun_component,
                op=SolarExposure(num_times=num_times),
            )
            self.register_output('percent_exposed_area', percent_exposed_area)

    sim = Simulator(M())
    sim.check_partials(compact_print=True)
