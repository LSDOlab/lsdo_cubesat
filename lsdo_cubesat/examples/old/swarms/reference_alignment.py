from csdl_om import Simulator

from lsdo_cubesat.orbit.reference_orbit_group import ReferenceOrbit
from lsdo_cubesat.telescope.telescope_configuration import TelescopeConfiguration
from lsdo_cubesat.examples.swarms.swarm1 import swarm
from csdl import Model


class Test(Model):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('step_size', types=float)
        self.parameters.declare('swarm')

    def define(self):
        swarm = self.parameters['swarm']
        num_times = swarm['num_times']
        step_size = swarm['step_size']
        self.add(
            ReferenceOrbit(
                num_times=num_times,
                step_size=step_size,
            ),
            name='reference_orbit',
        )
        self.add(
            TelescopeConfiguration(swarm=swarm),
            name='alignment',
        )


if __name__ == "__main__":
    # m.visualize_sparsity()
    sim = Simulator(Test(swarm=swarm))
    sim.visualize_implementation(recursive=True)
