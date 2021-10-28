import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent


class KSComp(ExplicitComponent):
    def initialize(self):
        """
        Declare options.
        """
        self.options.declare('in_name', types=str)
        self.options.declare('out_name', types=str)

        self.options.declare('shape', types=tuple)
        self.options.declare('constraint_size', types=int, default=1)

        self.options.declare('lower_flag', types=bool, default=False)
        self.options.declare('rho',
                             50.0,
                             desc="Constraint Aggregation Factor.")
        self.options.declare(
            'bound', 0.0, desc="Upper bound for constraint, default is zero.")

    def setup(self):
        """
        Declare inputs, outputs, and derivatives for the KS component.
        """
        in_name = self.options['in_name']
        out_name = self.options['out_name']

        shape = self.options['shape']
        constraint_size = self.options['constraint_size']

        # Inputs
        self.add_input(in_name, shape=shape + (constraint_size, ))

        # Outputs
        self.add_output(out_name, shape=shape)

        size = np.prod(shape)

        rows = np.zeros(constraint_size, dtype=np.int)
        cols = range(constraint_size)
        rows = np.tile(rows, size) + np.repeat(np.arange(size),
                                               constraint_size)
        cols = np.tile(cols, size) + np.repeat(
            np.arange(size), constraint_size) * constraint_size

        self.declare_partials(of=out_name, wrt=in_name, rows=rows, cols=cols)

    def compute(self, inputs, outputs):
        """
        Compute the output of the KS function.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        shape = self.options['shape']
        constraint_size = self.options['constraint_size']
        lower_flag = self.options['lower_flag']
        rho = self.options['rho']
        bound = self.options['bound']

        con_val = inputs[in_name] - bound
        if lower_flag:
            con_val = -con_val

        g_max = np.max(con_val, axis=-1)
        g_diff = con_val - np.einsum(
            '...,i->...i',
            g_max,
            np.ones(constraint_size),
        )
        exponents = np.exp(rho * g_diff)
        summation = np.sum(exponents, axis=-1)
        result = g_max + 1.0 / rho * np.log(summation)
        if lower_flag:
            result = -result
        outputs[out_name] = result

        dsum_dg = rho * exponents
        dKS_dsum = 1.0 / (rho * np.einsum(
            '...,i->...i',
            summation,
            np.ones(constraint_size),
        ))
        dKS_dg = dKS_dsum * dsum_dg

        self.dKS_dg = dKS_dg

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            unscaled, dimensional input variables read via inputs[key]
        partials : Jacobian
            sub-jac components written to partials[output_name, input_name]
        """
        in_name = self.options['in_name']
        out_name = self.options['out_name']
        partials[out_name, in_name] = self.dKS_dg.flatten()


if __name__ == '__main__':
    from openmdao.api import Problem, IndepVarComp

    shape = (4, 1, 3)
    # shape = (1,)
    constraint_size = 1

    prob = Problem()

    comp = IndepVarComp()
    comp.add_output('x', val=np.random.rand(*shape, constraint_size))
    # comp.add_output('x', val=np.ones((*shape, constraint_size)))
    prob.model.add('ivc', comp, promotes=['*'])

    comp = KSComp(
        in_name='x',
        out_name='y',
        shape=shape,
        constraint_size=1,
        lower_flag=False,
        rho=100.,
        bound=0.,
    )
    prob.model.add('comp', comp, promotes=['*'])

    prob.setup()
    prob.run_model()
    print(prob['x'])
    print(prob['x'].shape)
    print(prob['y'])
    print(prob['y'].shape)
    prob.check_partials(compact_print=True)
    # print(prob['x'], 'x')
    # print(prob['y'], 'y')
