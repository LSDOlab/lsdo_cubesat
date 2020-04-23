import numpy as np
import openmdao.api as om

from openmdao.api import Group, IndepVarComp, NonlinearBlockGS, LinearBlockGS
from lsdo_utils.api import ArrayReorderComp, LinearCombinationComp, PowerCombinationComp, ScalarContractionComp


from lsdo_cubesat.communication.Comm_GSposEarth import GroundStationComp

from lsdo_cubesat.communication.Data_download_comp import DataDownloadComp





class CommunicationGroup(Group):
    
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('step_size', types=float)
        
    def setup(self):
        num_times = self.options['num_times']
        step_size = self.options['step_size']

        # Design Parameters
        
        comp = IndepVarComp()
#        comp.add_output('Data0',val= )
#        comp.add_output('antAngle',val= )
        comp.add_output('lon',val= )
        comp.add_output('lat',val= )
        comp.add_output('alt',val= )
#        comp.add_output('r_e2b_I',val= )
#        comp.add_output('O_BI',val= )
#        comp.add_output('t',val= )
#        self.add_subsystem('inputs_comp', comp, promotes=['*'])
        
        
        comp = GroundStationComp(num_times = num_times)
        self.add_subsystem('r_e2g_E',comp,promotes=['*'])
        
        
        
        
#        comp = BitRateComp(
#            num_times = num_times,
#        )
#        self.add_subsystem('Download_rate_comp',comp,promotes=['*'])
#
#        comp = DataDownloadComp(
#            num_times = num_times,
#            step_size = step_size,
#        )
#        self.add_subsystem('Data_download_rk4_comp',comp, promotes=['*'])
        
        
if __name__ == '__main__':

    from openmdao.api import Problem, Group, IndepVarComp

    group = Group()

    comp = IndepVarComp()
    n = 2
    h =

    dm_dt_dsc = np.random.rand(1, n)
    Mass0_dsc = np.random.rand(1)
    comp.add_output('n', val=n)
    comp.add_output('dm_dt_dsc', val=dm_dt_dsc)
    comp.add_output('Mass0_dsc', val=Mass0_dsc)
    group.add_subsystem('Inputcomp', comp, promotes=['*'])

    group.add_subsystem('Statecomp_Implicit',
                        CommunicationGroup(n=n),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup()
    prob.run_model()
    prob.model.list_outputs()

    prob.check_partials(compact_print=True)
            
            

