"""
Determine the Transmitter Gain
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse
from MBI import MBI

from openmdao.api import Group, IndepVarComp, ExecComp,ExplicitComponent


from lsdo_utils.api import ArrayExpansionComp, BsplineComp, PowerCombinationComp, LinearCombinationComp

from lsdo_cubesat.utils.mtx_vec_comp import MtxVecComp

def fixangles(n, azimuth0, elevation0):
    """
    Take an input azimuth and elevation angles in radians, and fix them to
    lie on [0, 2pi]. Also swap azimuth 180deg if elevation is over 90 deg.
    """
    azimuth, elevation = np.zeros(azimuth0.shape), np.zeros(elevation0.shape)

    for i in range(n):
        azimuth[i] = azimuth0[i] % (2*np.pi)
        elevation[i] = elevation0[i] % (2*np.pi)
        if elevation[i] > np.pi:
            elevation[i] = 2*np.pi - elevation[i]
            azimuth[i] = np.pi + azimuth[i]
            azimuth[i] = azimuth[i] % (2.0*np.pi)

    return azimuth, elevation


class TransGainComp(ExplicitComponent):

    rawGdata = np.genfromtxt('Gain.txt')
    rawGaindata  = (10 ** (rawGdata / 10.0)).reshape((361, 361), order='F')

    az = np.linspace(0, 2 * np.pi, 361)
    el = np.linspace(0, 2 * np.pi, 361)
    
    def initialize(self):
        self.options.declare('num_times', types=int)

        self.MBI = MBI(self.rawGaindata, [self.az, self.el], [15, 15], [4, 4])
        
    
    def setup(self):
        opts = self.options
        n = opts['num_times']
        
        self.x = np.zeros((n, 2), order = 'F')
        
        self.add_input('azimuthGS', np.zeros(n), units='rad',
                       desc='Azimuth angle from satellite to ground station in '
                            'Earth-fixed frame over time')

        self.add_input('elevationGS', np.zeros(n), units='rad',
                       desc='Elevation angle from satellite to ground station '
                            'in Earth-fixed frame over time')

        # Outputs
        self.add_output('gain', np.zeros(n), units=None,
                        desc='Transmitter gain over time')
                        
    def compute(self, inputs, outputs):
        num_times = self.options['num_times']
        
        result = fixangles(num_times, inputs['azimuthGS'], inputs['elevationGS'])
        self.x[:,0] = result[0]
        self.x[:,0] = result[1]
        outputs['gain'] = self.MBI.evaluate(self.x)[:,0]
        
    def compute_partials(self, inputs, partials):
        
        self.dg_daz = self.MBI.evaluate(self.x, 1)[:, 0]
        self.dg_del = self.MBI.evaluate(self.x, 2)[:, 0]
        

if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp


    n = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('azimuthGS',val=np.random.random(n))
    comp.add_output('elevationGS',val=np.random.random(n))
    prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])

    comp = TransGainComp(
        num_times=n,
    )
    prob.model.add_subsystem('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)



