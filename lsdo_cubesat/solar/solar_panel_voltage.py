import numpy as np
from csdl import CustomExplicitOperation

from smt.surrogate_models import RMTB, RMTC


class SolarPanelVoltage(CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_times', default=1, types=int)

    def setup(self):
        n = self.parameters['num_times']
        self.add_input('temperature', shape=(n, ))
        self.add_input('sunlit_area', shape=(n, ))
        self.add_input('current', shape=(n, ))
        self.add_output('voltage', shape=(n, ))

        self.declare_partials(
            'voltage',
            ['temperature', 'sunlit_area', 'current'],
            rows=np.arange(n),
            cols=np.arange(n),
        )

        # build surrogate model
        home = '/Users/victor/'
        dat = np.genfromtxt(home + 'packages/Battery-ECM/cadre_iv_curve.dat',
                            delimiter='\n')
        nT, nA, nI = dat[:3]
        nT = int(nT)
        nA = int(nA)
        nI = int(nI)
        T = dat[3:3 + nT]
        A = dat[3 + nT:3 + nT + nA]
        I = dat[3 + nT + nA:3 + nT + nA + nI]
        V = dat[3 + nT + nA + nI:].reshape((nT, nA, nI), order='F')

        xt, xlimits = structure_data([T, A, I], V)

        self.sm = RMTB(
            xlimits=xlimits,
            order=(3, 3, 3),
            num_ctrl_pts=(6, 6, 15),
            energy_weight=1e-15,
            regularization_weight=0.0,
            print_global=False,
        )
        self.sm.set_training_values(xt, V.flatten())
        self.sm.train()
        self.x = np.zeros(3 * n).reshape((n, 3))

    def compute(self, inputs, outputs):
        t = inputs['temperature']
        a = inputs['sunlit_area']
        i = inputs['current']
        for idx in range(n):
            tai = np.array([t[idx], a[idx], i[idx]]).reshape((1, 3))
            outputs['voltage'][idx] = self.sm.predict_values(tai)

    def compute_partials(self, inputs, partials):
        t = inputs['temperature']
        a = inputs['sunlit_area']
        i = inputs['current']
        for idx in range(n):
            tai = np.array([t[idx], a[idx], i[idx]]).reshape((1, 3))
            partials['voltage',
                     'temperature'][idx] = self.sm.predict_derivatives(
                         tai, 0).flatten()
            partials['voltage',
                     'sunlit_area'][idx] = self.sm.predict_derivatives(
                         tai, 1).flatten()
            partials['voltage', 'current'][idx] = self.sm.predict_derivatives(
                tai, 2).flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp, Model

    n = 30
    indep = IndepVarComp()
    indep.add_output('temperature', val=np.random.rand(n))
    indep.add_output('sunlit_area', val=np.random.rand(n))
    indep.add_output('current', val=np.random.rand(n))
    prob = Problem()
    prob.model = Model()
    prob.model.add(
        'indeps',
        indep,
    )
    prob.model.add(
        'spv',
        SolarPanelVoltage(num_times=n),
    )

    prob.setup(check=True, force_alloc_complex=True)
    prob.check_partials(compact_print=True)
