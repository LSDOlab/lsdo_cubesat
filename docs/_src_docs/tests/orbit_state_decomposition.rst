Orbit State Decomposition
=========================

.. code-block:: python

  from openmdao.api import Problem, IndepVarComp
  from lsdo_cubesat.orbit.orbit_state_decomposition_comp import OrbitStateDecompositionComp
  import numpy as np
  
  num_times = 3
  
  prob = Problem()
  comp = IndepVarComp()
  comp.add_output('orbit_state', np.random.rand(6, num_times))
  prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])
  
  comp = OrbitStateDecompositionComp(num_times=num_times,
                                     orbit_state_name='orbit_state',
                                     position_name='position',
                                     velocity_name='velocity')
  prob.model.add_subsystem('comp', comp, promotes=['*'])
  
  prob.setup(check=True)
  cp = prob.check_partials(compact_print=True)
  
  
  # pytest
  def test_fn():
      from openmdao.utils.assert_utils import assert_check_partials
      assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)
  
::

  INFO: checking out_of_order
  INFO: checking system
  INFO: checking solvers
  INFO: checking dup_inputs
  INFO: checking missing_recorders
  WARNING: The Problem has no recorder of any kind attached
  INFO: checking comp_has_no_outputs
  ---------------------------------------------
  Component: OrbitStateDecompositionComp 'comp'
  ---------------------------------------------
  '<output>' wrt '<variable>'  | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  --------------------------------------------------------------------------------
  
  'position' wrt 'orbit_state' | 3.0000e+00 | 3.0000e+00 | 1.7703e-10 | 5.9011e-11
  'velocity' wrt 'orbit_state' | 3.0000e+00 | 3.0000e+00 | 2.0580e-10 | 6.8598e-11
  
  ############################################################################
  Sub Jacobian with Largest Relative Error: OrbitStateDecompositionComp 'comp'
  ############################################################################
  '<output>' wrt '<variable>'  | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  --------------------------------------------------------------------------------
  'velocity' wrt 'orbit_state' | 3.0000e+00 | 3.0000e+00 | 2.0580e-10 | 6.8598e-11
  
