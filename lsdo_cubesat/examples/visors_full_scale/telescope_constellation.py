from csdl import Model
import csdl
import numpy as np

from lsdo_cubesat.telescope.telescope_configuration import TelescopeConfiguration
from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.examples.visors_baseline.cubesat_group import Cubesat
from lsdo_cubesat.examples.visors_baseline.telescope import Telescope

# TODO: In run.py, create instance of SwarmParams defining constellation
# of telescopes; requires setting orbit parameters for each cubesat
# TODO: define TelescopeConstellation class using CSDL (below)


class TelescopeConstellation(Model):

    def initialize(self):
        self.parameters.declare('swarm', types=SwarmParams)

    def define(self):
        swarm = self.parameters['swarm']

        objectives = []
        for telescope_params in swarm.children:
            name = telescope_params.name
            self.add(
                Telescope(telescope_params),
                name=name,
                promotes=[],
            )
            # TODO: find other relevant connections to maintain even
            # intervals between telescopes
            obj = self.declare_variable('obj')
            self.connect('{}.obj'.format(name), 'obj')

        # TODO: get reasonable coefficients for regularization term
        obj = csdl.sum(*objectives)
        self.register_output('obj', obj)
        self.add_objective('obj')
