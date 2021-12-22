"""
    Coordinate transformation from the body frame to the inertial frame.
"""

import numpy as np

from csdl import CustomExplicitOperation


class RotMtxBIComp(CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_times', types=int)

    def define(self):
        num_times = self.parameters['num_times']

        self.add_input('roll', shape=num_times)
        self.add_input('pitch', shape=num_times)
        self.add_output('rot_mtx_b_i_3x3xn', shape=(3, 3, num_times))

        rows = np.arange(9 * num_times)
        cols = np.outer(
            np.ones(9, int),
            np.arange(num_times),
        ).flatten()
        self.declare_derivatives('rot_mtx_b_i_3x3xn',
                                 'roll',
                                 rows=rows,
                                 cols=cols)
        self.declare_derivatives('rot_mtx_b_i_3x3xn',
                                 'pitch',
                                 rows=rows,
                                 cols=cols)

    def compute(self, inputs, outputs):
        roll = inputs['roll']
        pitch = inputs['pitch']

        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)

        outputs['rot_mtx_b_i_3x3xn'][0, 0, :] = cos_roll
        outputs['rot_mtx_b_i_3x3xn'][0, 1, :] = sin_roll * cos_pitch
        outputs['rot_mtx_b_i_3x3xn'][0, 2, :] = sin_roll * sin_pitch
        outputs['rot_mtx_b_i_3x3xn'][1, 0, :] = -sin_roll
        outputs['rot_mtx_b_i_3x3xn'][1, 1, :] = cos_roll * cos_pitch
        outputs['rot_mtx_b_i_3x3xn'][1, 2, :] = cos_roll * sin_pitch
        outputs['rot_mtx_b_i_3x3xn'][2, 1, :] = -sin_pitch
        outputs['rot_mtx_b_i_3x3xn'][2, 2, :] = cos_pitch
        print(outputs['rot_mtx_b_i_3x3xn'].shape)

    def compute_derivatives(self, inputs, partials):
        num_times = self.parameters['num_times']

        roll = inputs['roll']
        pitch = inputs['pitch']

        cos_roll = np.cos(roll)
        sin_roll = np.sin(roll)
        cos_pitch = np.cos(pitch)
        sin_pitch = np.sin(pitch)

        dmtx_droll = partials['rot_mtx_b_i_3x3xn', 'roll'].reshape(
            (3, 3, num_times))
        dmtx_dpitch = partials['rot_mtx_b_i_3x3xn', 'pitch'].reshape(
            (3, 3, num_times))

        dmtx_droll[0, 0, :] = -sin_roll
        dmtx_droll[0, 1, :] = cos_roll * cos_pitch
        dmtx_droll[0, 2, :] = cos_roll * sin_pitch
        dmtx_droll[1, 0, :] = -cos_roll
        dmtx_droll[1, 1, :] = -sin_roll * cos_pitch
        dmtx_droll[1, 2, :] = -sin_roll * sin_pitch

        dmtx_dpitch[0, 1, :] = -sin_roll * sin_pitch
        dmtx_dpitch[0, 2, :] = sin_roll * cos_pitch
        dmtx_dpitch[1, 1, :] = -cos_roll * sin_pitch
        dmtx_dpitch[1, 2, :] = cos_roll * cos_pitch
        dmtx_dpitch[2, 1, :] = -cos_pitch
        dmtx_dpitch[2, 2, :] = -sin_pitch


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    num_times = 3

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('roll', val=np.random.random(num_times))
    comp.add_output('pitch', val=np.random.random(num_times))
    prob.model.add('inputs_comp', comp, promotes=['*'])

    comp = RotMtxBIComp(num_times=num_times, )
    prob.model.add('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    prob.model.list_outputs()
    prob.check_partials(compact_print=True)
