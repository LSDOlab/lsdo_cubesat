"""
Determine the Satellite Data Download Rate
"""

import numpy as np
from csdl import CustomExplicitOperation
from six.moves import range


class BitRate(CustomExplicitOperation):

    # constants
    pi = 2 * np.arccos(0.)
    c = 299792458
    Gr = 10**(12.9 / 10.)
    Ll = 10**(-2.0 / 10.)
    f = 437e6
    k = 1.3806503e-23
    SNR = 10**(5.0 / 10.)
    T = 500.
    alpha = c**2 * Gr * Ll / 16.0 / pi**2 / f**2 / k / SNR / T / 1e6

    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']
        # Inputs
        self.add_input('P_comm',
                       shape=num_times,
                       units='W',
                       desc='Communication power over time')

        self.add_input('gain',
                       shape=num_times,
                       units=None,
                       desc='Transmitter gain over time')

        self.add_input(
            'GSdist',
            shape=num_times,
            units='km',
            desc='Distance from ground station to satellite over time')

        self.add_input(
            'CommLOS',
            shape=num_times,
            units=None,
            desc='Satellite to ground station line of sight over time')

        # Outputs
        self.add_output('Download_rate',
                        shape=num_times,
                        units='Gibyte/s',
                        desc='Download rate over time')

    def compute(self, inputs, outputs):
        num_times = self.parameters['num_times']
        P_comm = inputs['P_comm']
        gain = inputs['gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']

        a = np.where(np.abs(GSdist > 1e-10))
        b = np.where(np.abs(GSdist <= 1e-10))
        S2 = np.zeros(num_times)
        S2[a] = GSdist[a] * 1e3
        S2[b] = 1e-10
        outputs['Download_rate'] = self.alpha * P_comm * gain * \
            CommLOS / S2 ** 2

        # for i in range(0, num_times):
        #     if np.abs(GSdist[i]) > 1e-10:
        #         S2 = GSdist[i] * 1e3
        #     else:
        #         S2 = 1e-10
        #     outputs['Download_rate'][i] = self.alpha * P_comm[i] * gain[i] * \
        #         CommLOS[i] / S2 ** 2

        # np.savetxt("rundata/GSdist.csv", GSdist, header="GSdist")
        # np.savetxt("rundata/P_comm.csv", P_comm, header="P_comm")
        # np.savetxt("rundata/gain.csv", gain, header="gain")
        # np.savetxt("rundata/CommLOS_final.csv", CommLOS, header="CommLOS")

        # Bitrate = outputs['Download_rate']
        # np.savetxt("rundata/Bitrate.csv", Bitrate, header="Bitrate")

    def compute_derivatives(self, inputs, partials):
        num_times = self.parameters["num_times"]

        P_comm = inputs['P_comm']
        gain = inputs['gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']

        S2 = 0.0
        self.dD_dP = np.zeros(num_times)
        self.dD_dGt = np.zeros(num_times)
        self.dD_dS = np.zeros(num_times)
        self.dD_dLOS = np.zeros(num_times)

        for i in range(0, num_times):

            if np.abs(GSdist[i]) > 1e-10:
                S2 = GSdist[i] * 1e3
            else:
                S2 = 1e-10

            self.dD_dP[i] = self.alpha * gain[i] * \
                CommLOS[i] / S2 ** 2
            self.dD_dGt[i] = self.alpha * P_comm[i] * \
                CommLOS[i] / S2 ** 2
            self.dD_dS[i] = -2.0 * 1e3 * self.alpha * P_comm[i] * \
                gain[i] * CommLOS[i] / S2 ** 3
            self.dD_dLOS[i] = self.alpha * \
                P_comm[i] * gain[i] / S2 ** 2

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        """
        Matrix-vector product with the Jacobian.
        """
        dDr = d_outputs['Download_rate']

        if mode == 'fwd':
            if 'P_comm' in d_inputs:
                dDr += self.dD_dP * d_inputs['P_comm']
            if 'gain' in d_inputs:
                dDr += self.dD_dGt * d_inputs['gain']
            if 'GSdist' in d_inputs:
                dDr += self.dD_dS * d_inputs['GSdist']
            if 'CommLOS' in d_inputs:
                dDr += self.dD_dLOS * d_inputs['CommLOS']
        else:
            if 'P_comm' in d_inputs:
                d_inputs['P_comm'] += self.dD_dP.T * dDr
            if 'gain' in d_inputs:
                d_inputs['gain'] += self.dD_dGt.T * dDr
            if 'GSdist' in d_inputs:
                d_inputs['GSdist'] += self.dD_dS.T * dDr
            if 'CommLOS' in d_inputs:
                d_inputs['CommLOS'] += self.dD_dLOS.T * dDr


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp
    import numpy as np

    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('P_comm', val=16.0 * np.ones(num_times), units='W')
    comp.add_output('gain', val=1.0 * np.ones(num_times))
    comp.add_output('GSdist', val=600.0 * np.ones(num_times), units='km')
    comp.add_output('CommLOS', val=np.ones(num_times))

    prob.model.add('inputs_comp', comp, promotes=['*'])

    comp = BitRateComp(num_times=num_times, )
    prob.model.add('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    print(prob['Download_rate'])
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
