import numpy as np
from csdl import CustomExplicitOperation

from smt.surrogate_models import RMTB


class SolarIllumination(CustomExplicitOperation):
    """
    Generic model for computing solar power as a function of azimuth and
    elevation (roll and pitch)
    """
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('sm', types=RMTB)

    def define(self):
        n = self.parameters['num_times']
        self.sm = self.parameters['sm']

        self.add_input('sun_direction', shape=(3, n))
        self.add_output('solar_illumination', shape=(n, ))
        self.declare_partials(
            'solar_illumination',
            'sun_direction',
            # rows=np.arange(n),
            # cols=np.arange(n),
        )

    def compute(self, inputs, outputs):
        sun_direction = inputs['sun_direction']

        # TODO: flatten?
        outputs['solar_illumination'] = self.sm.predict_values(sun_direction)

    def compute_partials(self, inputs, partials):
        sun_direction = inputs['sun_direction']

        partials['solar_illumination',
                 'sun_direction'] = self.sm.predict_derivatives(
                     sun_direction,
                     0,
                 ).flatten()
