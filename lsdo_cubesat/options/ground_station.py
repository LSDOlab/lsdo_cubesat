import numpy as np

from lsdo_cubesat.utils.api import OptionsDictionary


class GroundStation(OptionsDictionary):
    def initialize(self):
        self.declare('name', types=str)
        self.declare('lon', types=float)
        self.declare('lat', types=float)
        self.declare('alt', types=float)
        self.declare('antenna_angle', default=2., types=float)
