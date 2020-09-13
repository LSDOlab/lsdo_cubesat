import numpy as np
from openmdao.api import ExplicitComponent

from lsdo_cubesat.utils.structure_data import structure_data
from smt.surrogate_models import RMTB


class SolarExposure(ExplicitComponent):
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
        self.add_input('roll', shape=(n))
        self.add_input('pitch', shape=(n))
        self.add_output('sunlit_area', shape=(n))
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
        rp = np.concatenate(
            (
                r.reshape(num_times, 1),
                p.reshape(num_times, 1),
            ),
            axis=1,
        )
        outputs['sunlit_area'] = self.sm.predict_values(rp)

    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']
        r = inputs['roll']
        p = inputs['pitch']
        rp = np.concatenate(
            (
                r.reshape(num_times, 1),
                p.reshape(num_times, 1),
            ),
            axis=1,
        )
        partials['sunlit_area', 'roll'] = self.sm.predict_derivatives(
            rp,
            0,
        ).flatten()
        partials['sunlit_area', 'pitch'] = self.sm.predict_derivatives(
            rp,
            1,
        ).flatten()
        # for i in range(n):
        #     rp = np.array([r[i], p[i]]).reshape((1, 2))
        #     partials['sunlit_area', 'roll'][i] = self.sm.predict_derivatives(
        #         rp,
        #         0,
        #     )
        #     partials['sunlit_area', 'pitch'][i] = self.sm.predict_derivatives(
        #         rp,
        #         1,
        #     )


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    from lsdo_cubesat.solar.smt_exposure import smt_exposure
    np.random.seed(0)

    times = 200

    # load training data
    az = np.genfromtxt('lsdo_cubesat/data/arrow_xData.csv', delimiter=',')
    el = np.genfromtxt('lsdo_cubesat/data/arrow_yData.csv', delimiter=',')
    yt = np.genfromtxt('lsdo_cubesat/data/arrow_zData.csv', delimiter=',')

    # generate surrogate model with 20 training points
    # must be the same as the number of points used to create model
    sm = smt_exposure(20, az, el, yt)

    # check partials
    ivc = IndepVarComp()
    ivc.add_output('roll', val=np.random.rand(times))
    ivc.add_output('pitch', val=np.random.rand(times))
    prob = Problem()
    prob.model = Group()
    prob.model.add_subsystem(
        'ivc',
        ivc,
        promotes=['*'],
    )
    prob.model.add_subsystem(
        'spm',
        SolarExposure(num_times=times, sm=sm),
        promotes=['*'],
    )
    prob.setup()
    prob.check_partials(compact_print=True)
