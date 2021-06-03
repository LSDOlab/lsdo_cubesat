import numpy as np

from omtools.api import Group


class MaskVecGroup(Group):
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('swarm')

    def setup(self):
        num_times = self.options['num_times']

        self.register_output('mask_vec', num_times)

        self.declare_input('observation_dot', num_times)
        
#        self.declare_partials('mask_vec', 'observation_dot', val=0.)

#    def compute(self, inputs, outputs):
#        swarm = self.options['swarm']

#       outputs['mask_vec'] = 0.
#        outputs['mask_vec'][
#            inputs['observation_dot'] > swarm['cross_threshold']] = 1.
