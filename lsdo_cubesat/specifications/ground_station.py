import numpy as np

from lsdo_cubesat.utils.options_dictionary import OptionsDictionary


class GroundStationSpec(OptionsDictionary):

    def initialize(self):
        self.declare('name', types=str)
        self.declare('lon', types=float)
        self.declare('lat', types=float)
        self.declare('alt', types=float)
