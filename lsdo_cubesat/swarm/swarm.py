from lsdo_utils.api import OptionsDictionary


class Swarm(OptionsDictionary):

    def initialize(self):
        self.declare('num_times', types=int)
        self.declare('num_cp', types=int)
        self.declare('step_size', types=float)

        self.declare('cross_threshold', default=0.94, types=float)
        self.declare('launch_date', default=0., types=float)