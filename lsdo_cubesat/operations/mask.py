import numpy as np

from csdl import CustomExplicitOperation


class Mask(CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare('num_times', types=int)
        self.parameters.declare('threshold')
        self.parameters.declare('in_name')
        self.parameters.declare('out_name')

    def define(self):
        num_times = self.parameters['num_times']
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        self.add_input(in_name, shape=num_times)
        self.add_output(out_name, shape=num_times)
        self.declare_derivatives(out_name, in_name, val=0.)

    def compute(self, inputs, outputs):
        threshold = self.parameters['threshold']
        in_name = self.parameters['in_name']
        out_name = self.parameters['out_name']

        outputs[out_name] = 0.
        outputs[out_name][inputs[in_name] > threshold] = 1.
