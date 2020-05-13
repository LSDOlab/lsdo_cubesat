import numpy as np

from openmdao.api import ExplicitComponent


class CrossProductComp(ExplicitComponent):
    def initialize(self):
        self.options.declare('n', types=int)
        self.options.declare('first', types=str)
        self.options.declare('second', types=str)
        self.options.declare('out_name', types=str)

    def setup(self):
        n = self.options['n']
        first = self.options['first']
        second = self.options['second']
        out_name = self.options['out_name']
        self.add_input(first, shape=(3, n))
        self.add_input(second, shape=(3, n))
        self.add_output(out_name, shape=(3, n))

        self.declare_partials(out_name, first, method='cs')
        self.declare_partials(out_name, second, method='cs')

    def compute(self, inputs, outputs):
        first = inputs[self.options['first']]
        second = inputs[self.options['second']]
        c = outputs[self.options['out_name']]

        c[0, :] = first[1, :] * second[2, :] - first[2, :] * second[1, :]
        c[1, :] = first[2, :] * second[0, :] - first[0, :] * second[2, :]
        c[1, :] = first[0, :] * second[1, :] - first[1, :] * second[0, :]

    # def compute_partials(self, inputs, partials):
    #     first = inputs[self.options['first']]
    #     second = inputs[self.options['second']]
