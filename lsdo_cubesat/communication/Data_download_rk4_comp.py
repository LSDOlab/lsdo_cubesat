"""
RK4 component for Data Download
"""
import os
from six.moves import range

import numpy as np
import scipy.sparse

from openmdao.api import ExplicitComponent
from lsdo_cubesat.utils.rk4_comp import RK4Comp


class DataDownloadComp(RK4Comp):
    def setup(self):
        opts = self.options
        n = opts['num_times']
        h = opts['step_size']

        self.add_input('KS_Download_rate',
                       np.zeros((1, n)),
                       desc='Data download rate over time')

        self.add_input('initial_data',
                       np.zeros((1, )),
                       desc='Initial download data state')

        #States
        self.add_output('Data',
                        np.zeros((1, n)),
                        desc='Download data state over time')

        self.options['state_var'] = 'Data'
        self.options['init_state_var'] = 'initial_data'
        self.options['external_vars'] = ['KS_Download_rate']

        self.dfdy = np.array([[0.]])
        self.dfdx = np.array([[1.]])

    def f_dot(self, external, state):
        return external[0]

    def df_dy(self, external, state):
        return self.dfdy

    def df_dx(self, external, state):
        return self.dfdx


if __name__ == '__main__':

    from openmdao.api import Problem, Group
    from openmdao.api import IndepVarComp

    group = Group()

    comp = IndepVarComp()
    num_times = 1501
    num_cp = 300
    step_size = 95 * 60 / (num_times - 1)

    dd_dt = np.loadtxt('/home/lsdo/Cubesat/lsdo_cubesat/rundata/Bitrate.csv')
    # dd_dt.reshape((1, num_times))
    print(dd_dt.shape)
    Data0 = 0
    comp.add_output('num_times', val=num_times)
    comp.add_output('Download_rate', val=dd_dt)
    comp.add_output('initial_data', val=Data0)

    group.add_subsystem('Inputcomp', comp, promotes=['*'])

    group.add_subsystem('Statecomp_Implicit',
                        DataDownloadComp(num_times=num_times,
                                         step_size=step_size),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    print(prob['Data'])
    prob.check_partials(compact_print=True)

    # import matplotlib.pyplot as plt
    # time = np.arange(num_times)
    # Data = prob["Data"]
    # Data.reshape((num_times, 1))
    # print(Data.shape)
    # print(time.shape)
    # plt.plot(time, dd_dt, label="Datarate")
    # plt.plot(time, Data.T, label='Data')
    # plt.show()
