"""
Determine the Satellite Data Download Rate
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse

from openmdao.api import Group, IndepVarComp, ExecComp, ExplicitComponent

from lsdo_utils.api import ArrayExpansionComp, BsplineComp, PowerCombinationComp, LinearCombinationComp

from lsdo_cubesat.utils.mtx_vec_comp import MtxVecComp


# from lsdo_cubesat.communication.DataDownloadComp import DataDownloadComp


class BitRateComp(ExplicitComponent):

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
        num_times = self.options['num_times']

        self.add_input('P_comm', shape=num_times, units='W',
                       desc='Communication power over time')

        self.add_input('Gain', shape=num_times, units=None,
                       desc='Transmitter gain over time')

        self.add_input('GSdist', shape=num_times, units='km',
                       desc='Distance from ground station to satellite over time')

        self.add_input('CommLOS', shape=num_times, units=None,
                       desc='Satellite to ground station line of sight over time')

        self.add_output('Download_rate', shape=num_times)

        rows = np.arange(num_times).flatten()
        cols = np.arange(num_times).flatten()
        self.declare_partials('Download_rate','P_comm',rows=rows,cols=cols)
        self.declare_partials('Download_rate','Gain',rows=rows,cols=cols)
        self.declare_partials('Download_rate','GSdist',rows=rows,cols=cols)
        self.declare_partials('Download_rate','CommLOS',rows=rows,cols=cols)

    def compute(self, inputs, outputs):
        num_times = self.options['num_times']

        outputs['Download_rate'] = self.alpha * inputs['P_comm'] * inputs['Gain'] * inputs['CommLOS']\
                                   /inputs['GSdist'] ** 2


    def compute_partials(self, inputs, partials):
        num_times = self.options['num_times']

        P_comm = inputs['P_comm']
        Gain = inputs['Gain']
        GSdist = inputs['GSdist']
        CommLOS = inputs['CommLOS']

        dD_dP = partials['Download_rate','P_comm'].reshape((num_times,1))
        dD_dGt = partials['Download_rate','Gain'].reshape((num_times,1))
        dD_dS = partials['Download_rate','GSdist'].reshape((num_times,1))
        dD_dLOS = partials['Download_rate','CommLOS'].reshape((num_times,1))


        dD_dP[:,0] = self.alpha * Gain * CommLOS / GSdist ** 2
        dD_dGt[:,0] = self.alpha * P_comm * CommLOS / GSdist ** 2
        dD_dS[:,0] = -2.0 * self.alpha * P_comm * Gain * CommLOS / GSdist ** 3
        dD_dLOS[:,0] = self.alpha * P_comm * Gain / GSdist ** 2



if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('P_comm', val=np.random.random(num_times),units='W')
    comp.add_output('Gain', val=np.random.random(num_times))
    comp.add_output('GSdist', val=np.random.random(num_times),units='km')
    comp.add_output('CommLOS', val=np.random.random(num_times))

    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = BitRateComp(
        num_times=num_times,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)


