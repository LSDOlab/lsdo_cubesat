import numpy as np
from openmdao.api import ExplicitComponent

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
        # atan2 -> (-pi, pi)
        # asin -> (-pi/2, pi/2)
        # acos -> (0, pi)
        r = inputs['roll']
        p = inputs['pitch']

        # r = np.sign(r) * np.mod(r, np.pi)
        # p = np.sign(p) * np.mod(p, np.pi / 2)

        # quitfn = False
        # if np.any(r > np.pi) or np.any(np.any(r < -np.pi)):
        #     print('ROLL OUT OF BOUNDS')
        #     print(np.amin(r))
        #     print(np.amax(r))
        #     print(r)
        #     quitfn = True
        # if np.any(p > np.pi / 2) or np.any(np.any(p < -np.pi / 2)):
        #     print('PITCH OUT OF BOUNDS')
        #     print(np.amin(p))
        #     print(np.amax(p))
        #     print(p)
        #     quitfn = True
        # if quitfn:
        #     exit()
        rp = np.concatenate(
            (
                r.reshape(num_times, 1),
                p.reshape(num_times, 1),
            ),
            axis=1,
        )
        self.a = np.maximum(self.sm.predict_values(rp), 0)
        outputs['sunlit_area'] = self.a
        # outputs['sunlit_area'] = self.sm.predict_values(rp)
        # if np.any(outputs['sunlit_area'] >= 1) or np.any(
        #         outputs['sunlit_area'] < 0):
        #     print('SUNLIT AREA OUT OF BOUNDS')
        #     print(outputs['sunlit_area'])
        # exit()

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

        ar = self.sm.predict_derivatives(
            rp,
            0,
        )
        ind = np.where(self.a < 0)
        if len(ind) > 0:
            ar[ind] = 0
        partials['sunlit_area', 'roll'] = ar.flatten()

        ap = self.sm.predict_derivatives(
            rp,
            1,
        )
        ind = np.where(self.a < 0)
        if len(ind) > 0:
            ap[ind] = 0
        partials['sunlit_area', 'pitch'] = ap.flatten()


if __name__ == "__main__":
    from openmdao.api import Problem, IndepVarComp, Group
    from lsdo_cubesat.solar.smt_exposure import smt_exposure
    from lsdo_cubesat.utils.random_arrays import make_random_bounded_array
    np.random.seed(0)

    times = 200

    # load training data
    az = np.genfromtxt('lsdo_cubesat/data/arrow_xData.csv', delimiter=',')
    el = np.genfromtxt('lsdo_cubesat/data/arrow_yData.csv', delimiter=',')
    yt = np.genfromtxt('lsdo_cubesat/data/arrow_zData.csv', delimiter=',')

    # generate surrogate model with 20 training points
    # must be the same as the number of points used to create model
    sm = smt_exposure(20, az, el, yt)

    roll = make_random_bounded_array(times, np.pi)
    pitch = make_random_bounded_array(times, np.pi / 2)
    print(roll)
    print(pitch)
    print(np.amin(roll))
    print(np.amin(pitch))
    print(np.amax(roll))
    print(np.amax(pitch))

    # check partials
    ivc = IndepVarComp()
    ivc.add_output('roll', val=roll)
    ivc.add_output('pitch', val=pitch)
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
