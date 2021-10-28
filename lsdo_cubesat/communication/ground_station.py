import numpy as np

from lsdo_cubesat.utils.options_dictionary import OptionsDictionary


class GroundStationParams(OptionsDictionary):
    def initialize(self):
        self.declare('name', types=str)
        self.declare('lon', types=float)
        self.declare('lat', types=float)
        self.declare('alt', types=float)
        self.declare('antAngle', default=2., types=float)
