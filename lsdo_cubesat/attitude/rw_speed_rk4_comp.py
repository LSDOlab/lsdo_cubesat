import numpy as np
from lsdo_cubesat.lsdo_cubesat.utils.rk4_comp import RK4Comp

# 1 - adapt component defined below to use three torques as inputs
# code below only uses one torque
# use other RK4Comp classes as a guide for indices in f_dot, and df_dx methods
# 2 - add component to attitude model
# 3 - using omtools, compute torques

# attitude group should have code that looks like this:
# self.add_subsystem('reaction_wheel_speed', RWSpeedRK4Comp, promotes=['*'])
# rw_speed = self.declare_input('rw_speed', shape=(n,))
# power = self.declare_input('power', shape=(n,))
# torque = power / rw_speed
# self.register_output('torque', torque)
# self.add_constraint('rw_speed', upper=250)
# self.add_constraint('rw_speed', lower=-250)
# rename variables accordingly
# we'll add some constraints to prevent saturation


class RWSpeedRK4Comp(RK4Comp):
    """
    Inherit from this component to use.
​
    State variable dimension: (num_states, num_time_points)
    External input dimension: (input width, num_time_points)
    """
    def initialize(self):
        self.options.declare('num_times', types=int)
        self.options.declare('step_size', types=float)
        # moment of inertia for BCT RwP015 reaction wheel
        self.options.declare('mmoi', types=float, default=3e-5)
        self.options['external_vars'] = [
            'torque_x',
            'torque_y',
            'torque_z',
        ]
        self.options['init_state_var'] = 'initial_accel'
        self.options['state_var'] = 'accel'

    def setup(self):
        n = self.options['num_times']
        self.add_input(
            'torque_x',
            val=0,
            shape=(n, ),
            desc='torque applied to reaction wheel in the x-direction',
        )
        self.add_input(
            'torque_y',
            val=0,
            shape=(n, ),
            desc='torque applied to reaction wheel in the y-direction',
        )
        self.add_input(
            'torque_z',
            val=0,
            shape=(n, ),
            desc='torque applied to reaction wheel in the z-direction',
        )
        self.add_input(
            'initial_accel',
            val=0,
            shape=(3, ),
            desc='initial angular acceleration of reaction wheel',
        )
        self.add_output(
            'accel',
            val=0,
            shape=(3, n),
            desc='angular acceleration of reaction wheel',
        )

    # specify ODE
    def f_dot(self, external, state):
        J = self.options['mmoi']
        state_dot = np.zeros(3)

        for i in range(3):
            state_dot[i] = external[i] / J

        return state_dot

    # ODE wrt state variables
    def df_dy(self, external, state):

        return np.zeros((3, 3))

    # ODE wrt external inputs
    def df_dx(self, external, state):
        J = self.options['mmoi']

        dfdx = np.zeros((3, 3))
        dfdx[0, 0] = 1.0
        dfdx[1, 1] = 1.0
        dfdx[2, 2] = 1.0

        return dfdx / J


if __name__ == '__main__':
    state = (3, 1)
    external = (0, 1, 2)

    rw_speed = RWSpeedRK4Comp()
    dfdt = rw_speed.f_dot(external, state)
    print(dfdt)
    dfdy = rw_speed.df_dy(external, state)
    print(dfdy)
    dfdx = rw_speed.df_dx(external, state)
    print(dfdx)
