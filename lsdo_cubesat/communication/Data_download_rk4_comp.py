"""
RK4 component for Data Download
"""

import numpy as np

from lsdo_cubesat.operations.rk4_op import RK4


class DataDownloadComp(RK4):

    def define(self):
        opts = self.parameters
        n = opts['num_times']
        h = opts['step_size']

        self.add_input('KS_Download_rate',
                       np.zeros((1, n)),
                       desc='Data download rate over time')

        self.add_input('Initial_Data',
                       np.zeros((1, )),
                       desc='Initial download data state')

        #States
        self.add_output('Data',
                        np.zeros((1, n)),
                        desc='Download data state over time')

        self.parameters['state_var'] = 'Data'
        self.parameters['init_state_var'] = 'Initial_Data'
        self.parameters['external_vars'] = ['KS_Download_rate']

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
    comp.add_output('Initial_Data', val=Data0)

    group.add('Inputcomp', comp, promotes=['*'])

    group.add('Statecomp_Implicit',
              DataDownloadComp(num_times=num_times, step_size=step_size),
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
