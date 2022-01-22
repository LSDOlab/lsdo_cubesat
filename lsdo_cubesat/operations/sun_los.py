from csdl import CustomExplicitOperation
import numpy as np
from lsdo_cubesat.constants import RADII

R = RADII['Earth']


class SunLOS(CustomExplicitOperation):

    def initialize(self):
        super().initialize()
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('alfa', types=float, default=0.9)
        self.parameters.declare('R', types=float, default=R)

    def define(self):
        num_times = self.parameters['num_times']
        self.add_input('position_km', shape=(3, num_times))
        self.add_input('sun_direction', shape=(3, num_times))
        self.add_output('sun_LOS', shape=(1, num_times))

        # r = np.einsum('i,j->ij', np.arange(num_times), np.ones(3))
        # c = np.arange(3 * num_times)
        self.declare_derivatives(
            'sun_LOS',
            'position_km',
            # rows=r,
            # cols=c,
            method='cs',
        )
        self.declare_derivatives(
            'sun_LOS',
            'sun_direction',
            # rows=r,
            # cols=c,
            method='cs',
        )
        # self.dot = np.zeros(num_times).reshape((1,num_times))

    def compute(self, inputs, outputs):
        R = self.parameters['R']
        alfa = self.parameters['alfa']
        position_km = inputs['position_km']
        sun_direction = inputs['sun_direction']

        # component of position in direction of sun
        dot = np.sum(position_km * sun_direction, axis=(0, ))
        # distance from earth center projected onto plane normal to sun
        # direction
        ds = np.linalg.norm(
            np.cross(position_km, sun_direction, axis=0),
            axis=0,
        )

        # capture penumbra effect between light and shadow
        eta = (ds - alfa * R) / (R - alfa * R)

        outputs['sun_LOS'] = np.where(
            dot < 0.,
            np.where(
                ds > R,
                1.,
                np.where(ds > alfa * R, 3 * eta**2 - 2 * eta**2, 0),
            ),
            1.,
        )

    # def compute_derivatives(self, inputs, derivatives):
    #     mask = np.where(self.dot >= 0, 0, 1)
    #     derivatives['sun_LOS', 'position_km'] = 0
    #     derivatives['sun_LOS', 'sun_direction'] = 0


if __name__ == "__main__":
    from csdl import Model
    import csdl
    from csdl_om import Simulator

    class M(Model):

        def define(self):
            num_times = 5
            position_km = self.declare_variable(
                'position_km',
                shape=(3, num_times),
                val=np.random.rand(3 * num_times).reshape(
                    (3, num_times)) - 0.5,
            )
            sun_direction = self.declare_variable(
                'sun_direction',
                shape=(3, num_times),
                val=np.random.rand(3 * num_times).reshape(
                    (3, num_times)) - 0.5,
            )
            sun_LOS = csdl.custom(
                position_km,
                sun_direction,
                op=SunLOS(num_times=num_times),
            )
            self.register_output('sun_LOS', sun_LOS)

    sim = Simulator(M())
    sim.check_partials(compact_print=True)
