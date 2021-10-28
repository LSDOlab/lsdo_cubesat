from lsdo_cubesat.utils.options_dictionary import OptionsDictionary


class GS_net(OptionsDictionary):
    def initialize(self):

        self.declare('num_times', types=int)
        self.declare('num_cp', types=int)
        self.declare('step_size', types=float)
