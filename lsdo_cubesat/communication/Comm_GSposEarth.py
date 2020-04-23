import os
from six.moves import range

import numpy as np
import scipy.sparse


from openmdao.api import Group, IndepVarComp, ExecComp,ExplicitComponent


class GroundStationComp(ExplicitComponent):
    """
    Returns position of the ground station in Earth frame.
    """

    # Constants
    Re = 6378.137
    d2r = np.pi / 180.

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']
        # Inputs
        self.add_input('lon', 0.0, units='rad',
                       desc='Longitude of ground station in Earth-fixed frame')
        self.add_input('lat', 0.0, units='rad',
                       desc='Latitude of ground station in Earth-fixed frame')
        self.add_input('alt', 0.0, units='km',
                       desc='Altitude of ground station in Earth-fixed frame')

        # Outputs
        self.add_output('r_e2g_E', np.zeros((3, num_times)), units='km',
                        desc='Position vector from earth to ground station in '
                             'Earth-fixed frame over time')

    def compute(self, inputs, outputs):
        """
        Calculate outputs.
        """
        num_times = self.options['num_times']
        lat = inputs['lat']
        lon = inputs['lon']
        alt = inputs['alt']
        r_e2g_E = outputs['r_e2g_E']

        cos_lat = np.cos(self.d2r * lat)
        r_GS = (self.Re + alt)

        r_e2g_E[0, :] = r_GS * cos_lat * np.cos(self.d2r*lon)
        r_e2g_E[1, :] = r_GS * cos_lat * np.sin(self.d2r*lon)
        r_e2g_E[2, :] = r_GS * np.sin(self.d2r*lat)

    def compute_partials(self, inputs, partials):
        """
        Calculate and save derivatives. (i.e., Jacobian)
        """
        num_times = self.options['num_times']
        lat = inputs['lat']
        lon = inputs['lon']
        alt = inputs['alt']

        self.dr_dlon = np.zeros(3)
        self.dr_dlat = np.zeros(3)
        self.dr_dalt = np.zeros(3)

        cos_lat = np.cos(self.d2r * lat)
        sin_lat = np.sin(self.d2r * lat)
        cos_lon = np.cos(self.d2r * lon)
        sin_lon = np.sin(self.d2r * lon)

        r_GS = (self.Re + alt)

        self.dr_dlon[0] = -self.d2r * r_GS * cos_lat * sin_lon
        self.dr_dlat[0] = -self.d2r * r_GS * sin_lat * cos_lon
        self.dr_dalt[0] = cos_lat * cos_lon

        self.dr_dlon[1] = self.d2r * r_GS * cos_lat * cos_lon
        self.dr_dlat[1] = -self.d2r * r_GS * sin_lat * sin_lon
        self.dr_dalt[1] = cos_lat * sin_lon

        self.dr_dlon[2] = 0.
        self.dr_dlat[2] = self.d2r * r_GS * cos_lat
        self.dr_dalt[2] = sin_lat

if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    group = Group()
    comp = IndepVarComp()
    num_times = 3
    
    comp.add_output('lon',val=500.0)
    comp.add_output('lat',val=300.0)
    comp.add_output('alt',val=400.0)

    
    
#    comp.add_output('t', val=np.array([3600*2,3600*4,3600*6]))

    group.add_subsystem('Inputcomp', comp, promotes=['*'])
    group.add_subsystem('ground_station_position',
                        GroundStationComp(num_times=num_times),
                        promotes=['*'])

    prob = Problem()
    prob.model = group
    prob.setup(check=True)
    prob.run_model()
    prob.model.list_outputs()
#    print(prob['t'])
#    print(prob['q_E'])

    prob.check_partials(compact_print=True)
