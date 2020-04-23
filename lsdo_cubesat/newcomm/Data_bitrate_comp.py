"""
Determine the Satellite Data Download Rate
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse

from openmdao.api import Group, IndepVarComp, ExecComp,ExplicitComponent


from lsdo_utils.api import ArrayExpansionComp, BsplineComp, PowerCombinationComp, LinearCombinationComp

from lsdo_cubesat.utils.mtx_vec_comp import MtxVecComp
#from lsdo_cubesat.communication.DataDownloadComp import DataDownloadComp


class BitRateComp(ExplicitComponent):

    # Constant
    c = 299792458
    Gr = 10 ** (12.9 / 10.)
    Ll = 10 ** (-2.0 / 10.)
    f = 437e6
    k = 1.3806503e-23
    SNR = 10 ** (5.0 / 10.)
    T = 500.
    alpha = c ** 2 * Gr * Ll / 16.0 / np.pi ** 2 / f ** 2 / k / SNR / T / 1e6
    
    def initialize(self):
        self.options.declare('num_times', types=int)
    
    def setup(self):
        opts = self.options
        n = opts['num_times']
    
        # Inputs
        self.add_input('P_comm', np.zeros((1,n)), units='W',
                       desc='Communication power over time')

        self.add_input('Gain', np.zeros((1,n)), units=None,
                       desc='Transmitter gain over time')

        self.add_input('GSdist', np.zeros((1,n)), units='km',
                       desc='Distance from ground station to satellite over time')

        self.add_input('CommLOS', np.zeros((1,n)), units=None,
                       desc='Satellite to ground station line of sight over time')

        # Outputs
        self.add_output('Download_rate',np.zeros((1,n)))


    def compute(self,inputs,outputs):
        num_times = self.options['num_times']
    
        
        P_comm = inputs['P_comm']
        gain = inputs['Gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']
        Dr = outputs['Download_rate']
        
        for i in range(0, num_times):
            print(GSdist[i])
            if np.abs(GSdist[i]) > 1e-10:
                S2 = GSdist[i] * 1e3
            else:
                S2 = 1e-10
            Dr[i] = self.alpha * P_comm[i] * gain[i] * \
                CommLOS[i] / S2 ** 2
                
                
    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        
        num_times = self.options['num_times']
        
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

            if (np.abs(GSdist[i]) > 1e-10).any():
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
                
                
if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    n = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('P_comm',val=np.random.random((1,n)))
    comp.add_output('Gain',val=np.random.random((1,n)))
    comp.add_output('GSdist', val=np.random.random((1, n)))
    comp.add_output('CommLOS', val=np.zeros((1,n)))
    
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = BitRateComp(
        num_times=n,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
            
        
