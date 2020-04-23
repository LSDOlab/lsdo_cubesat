from openmdao.api import Group, IndepVarComp


class PreprocessGroup(Group):

    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('step_size', types=float)

    def setup(self):
        num_times = self.options['num_times']
        step_size = self.options['step_size']

        times = step_size * (num_times - 1)
        
        comp = IndepVarComp()
        comp.add_output('times', val=times)
        self.add_subsystem('inputs_comp', comp, promotes=['*'])