import numpy as np

from openmdao.api import ExplicitComponent

from rot_mtx_ECI_EF_comp import RotMtxECIEFComp



class GS_ECEF_Comp(ExplicitComponent):
    """
    Returns position of the ground station in Earth-fxied frame.
    """

    # Constants
    Re = 6378.137
    d2r = np.pi / 180.

    def initialize(self):
        self.options.declare('num_times', types=int)

    def setup(self):
        num_times = self.options['num_times']

        self.add_input('lon', 0.0, units='rad',
                       desc='Longitude of ground station in Earth-fixed frame')
        self.add_input('lat', 0.0, units='rad',
                       desc='Latitude of ground station in Earth-fixed frame')
        self.add_input('alt', 0.0, units='km',
                       desc='Altitude of ground station in Earth-fixed frame')

        self.add_output('r_e2g_E', np.zeros((3, num_times)), units='km',
                        desc='Position vector from earth to ground station in '
                             'Earth-fixed frame over time')

        self.declare_partials('r_e2g_E','lon',val=np.zeros((3,num_times)))
        self.declare_partials('r_e2g_E','lat',val=np.zeros((3,num_times)))
        self.declare_partials('r_e2g_E','alt',val=np.zeros((3,num_times)))

    def compute(self, inputs, outputs):
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
        num_times = self.options['num_times']
        lat = inputs['lat']
        lon = inputs['lon']
        alt = inputs['alt']

        dr_dlon = partials['r_e2g_E','lon'].reshape((3,num_times))
        dr_dlat = partials['r_e2g_E','lat'].reshape((3,num_times))
        dr_dalt = partials['r_e2g_E','alt'].reshape((3,num_times))

        cos_lat = np.cos(self.d2r * lat)
        sin_lat = np.sin(self.d2r * lat)
        cos_lon = np.cos(self.d2r * lon)
        sin_lon = np.sin(self.d2r * lon)

        r_GS = (self.Re + alt)

        dr_dlon[0,:] = -self.d2r * r_GS * cos_lat * sin_lon
        dr_dlat[0,:] = -self.d2r * r_GS * sin_lat * cos_lon
        dr_dalt[0,:] = cos_lat * cos_lon

        dr_dlon[1,:] = self.d2r * r_GS * cos_lat * cos_lon
        dr_dlat[1,:] = -self.d2r * r_GS * sin_lat * sin_lon
        dr_dalt[1,:] = cos_lat * sin_lon

        dr_dlon[2,:] = 0.0
        dr_dlat[2,:] = self.d2r * r_GS * cos_lat
        dr_dalt[2,:] = sin_lat




if __name__ == '__main__':
    import numpy as np

    from openmdao.api import Problem, IndepVarComp, Group

    num_times = 4

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('lon',val=-116.8873,units='rad')
    comp.add_output('lat',val=35.4227,units='rad')
    comp.add_output('alt',val=0.9,units='km')
    prob.model.add_subsystem('inputs_comp',comp,promotes=['*'])

    comp = GS_ECEF_Comp(num_times=num_times)
    prob.model.add_subsystem('comp',comp,promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)

