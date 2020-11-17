import numpy as np

from openmdao.api import ExplicitComponent


class OrbitStateDecompositionComp(ExplicitComponent):
    """
    Decompose state vector into position and velocity

    Options
    -------
    num_times : int
        Number of time steps in a given trajectory
    position_name : str
        Name of position vector
    velocity_name : str
        Name of velocity vector
    orbit_state_name : str
        Name of state vector to decompose; first three elements store
        position vector, last three store velocity vector
    """
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('position_name', types=str)
        self.options.declare('velocity_name', types=str)
        self.options.declare('orbit_state_name', types=str)

    def setup(self):
        num_times = self.options['num_times']
        position_name = self.options['position_name']
        velocity_name = self.options['velocity_name']
        orbit_state_name = self.options['orbit_state_name']

        self.add_input(orbit_state_name, shape=(6, num_times))
        self.add_output(position_name, shape=(3, num_times))
        self.add_output(velocity_name, shape=(3, num_times))

        orbit_state_indices = np.arange(6 * num_times).reshape((6, num_times))
        arange_3 = np.arange(3 * num_times)

        rows = arange_3
        cols = orbit_state_indices[:3, :].flatten()
        self.declare_partials(position_name,
                              orbit_state_name,
                              val=1.,
                              rows=rows,
                              cols=cols)

        rows = arange_3
        cols = orbit_state_indices[3:, :].flatten()
        self.declare_partials(velocity_name,
                              orbit_state_name,
                              val=1.,
                              rows=rows,
                              cols=cols)

    def compute(self, inputs, outputs):
        position_name = self.options['position_name']
        velocity_name = self.options['velocity_name']
        orbit_state_name = self.options['orbit_state_name']

        outputs[position_name] = inputs[orbit_state_name][:3, :]
        outputs[velocity_name] = inputs[orbit_state_name][3:, :]
