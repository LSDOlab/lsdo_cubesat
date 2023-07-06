import numpy as np
from openmdao.api import ExplicitComponent

from smt.surrogate_models import RMTB


class SolarIllumination(ExplicitComponent):
    """
    Generic model for computing solar power as a function of azimuth and
    elevation (roll and pitch)
    """
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('sm', types=RMTB)

    def setup(self):
        n = self.options['num_times']
        self.sm = self.options['sm']
        self.add_input('roll', shape=(n, ))
        self.add_input('pitch', shape=(n, ))
        self.add_output('sunlit_area', shape=(n, ))
        self.declare_partials(
            'sunlit_area',
            'roll',
            rows=np.arange(n),
            cols=np.arange(n),
        )
        self.declare_partials(
            'sunlit_area',
            'pitch',
            rows=np.arange(n),
            cols=np.arange(n),
        )

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']
        r = inputs['roll']
        p = inputs['pitch']

        # wrap input angles
        r = np.mod(r, np.pi)
        p = np.mod(p, np.pi)

        # predict sunlit area percent
        rp = np.concatenate(
            (
                r.reshape(num_times, 1),
                p.reshape(num_times, 1),
            ),
            axis=1,
        )
        predicted_area = self.sm.predict_values(rp)

        # ensure area percent between 0 and 1
        lower = np.min(predicted_area)
        upper = np.max(predicted_area)
        if lower < 0:
            shifted = predicted_area - lower
        else:
            shifted = predicted_area
        self.scaler = min(1, upper) / max(shifted)
        scaled = self.scaler * shifted
        outputs['sunlit_area'] = scaled.flatten()

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        r = inputs['roll']
        p = inputs['pitch']

        # wrap input angles
        r = np.mod(r, np.pi)
        p = np.mod(p, np.pi)

        # predict sunlit area percent derivatives
        rp = np.concatenate(
            (
                r.reshape(num_times, 1),
                p.reshape(num_times, 1),
            ),
            axis=1,
        )
        partials['sunlit_area', 'roll'] =\
            self.scaler * \
            self.sm.predict_derivatives(
            rp,
            0,
        ).flatten()
        partials['sunlit_area', 'pitch'] = \
            self.scaler * \
            self.sm.predict_derivatives(
            rp,
            1,
        ).flatten()