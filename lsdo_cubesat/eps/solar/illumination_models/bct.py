import numpy as np
import matplotlib.pyplot as plt

p = 1

n = 360 * p
m = n
azimuth = np.linspace(-np.pi, np.pi, m)
elevation = np.linspace(-np.pi / 2, np.pi / 2, n)

X, Y = np.meshgrid(azimuth, elevation)

# assume s/c fwd direction is perpendicular to solar panels
Ax = np.cos(X) * np.cos(Y)
Ay = np.sin(X) * np.cos(Y)
Az = np.sin(Y)

sc_to_sun = Ax / (Ax**2 + Ay**2 + Az**2)**(1 / 2)

w = 3
l = 3 * 3 * 2
A = w * l

illumination = np.where(sc_to_sun > 0, sc_to_sun * A, 0) / A

# fig = plt.figure()
# ax = fig.add_subplot()
# ax.contourf(X, Y, illumination)
# ax.set_xlabel('azimuth [rad]')
# ax.set_ylabel('elevation [rad]')
# plt.show()

from lsdo_cubesat.csdl_future.surrogate_models.rmtb import RMTB
# load training data
from lsdo_cubesat.solar.illumination_models.data.cubesat_xdata import cubesat_xdata as az
from lsdo_cubesat.solar.illumination_models.data.cubesat_ydata import cubesat_ydata as el
from lsdo_cubesat.solar.illumination_models.data.cubesat_zdata import cubesat_zdata as yt


class BCT(RMTB):

    def override_tuning_parameters(self):
        self.order = 4
        self.num_ctrl_pts = 50
        self.energy_weight = 1e-15
        self.regularization_weight = 1e-6

    def define_training_inputs(self):
        self.training_inputs['azimuth'] = azimuth[::p]
        self.training_inputs['elevation'] = elevation[::p]

    def define_training_outputs(self):
        self.training_outputs['illumination'] = illumination[::p, ::p]


if __name__ == '__main__':
    from csdl import Model, GraphRepresentation
    import csdl
    from python_csdl_backend import Simulator
    import matplotlib.pyplot as plt

    class M(Model):

        def initialize(self):
            self.parameters.declare('n', types=int)

        def define(self):
            n = self.parameters['n']

            azimuth = self.declare_variable('azimuth', shape=(n, ))
            elevation = self.declare_variable('elevation', shape=(n, ))
            illumination = csdl.custom(
                azimuth,
                elevation,
                op=BCT(shape=(n, ), ).create_op(),
            )
            self.register_output('illumination', illumination)

    # use a smaller number to check partials in less time
    dummy = BCT(shape=(n, ))
    ti = dummy.training_inputs
    to = dummy.training_outputs

    rep = GraphRepresentation(M(n=n))
    sim = Simulator(rep)
    np.random.seed(0)
    sim['azimuth'] = 2 * (np.random.rand(n) - 0.5) * np.pi
    sim['elevation'] = (np.random.rand(n) - 0.5) * np.pi
    sim.run()
    # sim.check_partials(compact_print=True)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X, Y, 100 * illumination)
    ax.scatter(sim['azimuth'], sim['elevation'], 100 * sim['illumination'])
    ax.set_xlabel('Azimuth')
    ax.set_ylabel('Elevation')
    ax.set_title('Percent Illumination')
    plt.show()
