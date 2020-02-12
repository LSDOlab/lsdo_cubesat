import numpy as np

from openmdao.api import ExplicitComponent


class MaskVecComp(ExplicitComponent):

    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('swarm')

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('observation_cross_norm', shape=num_times)
        self.add_output('mask_vec', shape=num_times)
        self.declare_partials('mask_vec', 'observation_cross_norm', val=0.)

    def compute(self, inputs, outputs):
        swarm = self.options['swarm']

        outputs['mask_vec'] = 0.
        outputs['mask_vec'][inputs['observation_cross_norm'] > swarm['cross_threshold']] = 1.