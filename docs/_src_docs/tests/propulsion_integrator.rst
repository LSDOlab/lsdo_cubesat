Mass Flow Rate Integrator
=========================

.. code-block:: python

  from openmdao.api import Problem
  from openmdao.api import IndepVarComp
  from lsdo_cubesat.propulsion.propellant_mass_rk4_comp import PropellantMassRK4Comp
  import numpy as np
  
  np.random.seed(0)
  
  comp = IndepVarComp()
  n = 20
  h = 6000.
  
  dm_dt = np.random.rand(1, n)
  Mass0 = np.random.rand(1)
  comp.add_output('num_times', val=n)
  comp.add_output('mass_flow_rate', val=dm_dt)
  comp.add_output('initial_propellant_mass', val=Mass0)
  prob = Problem()
  prob.model.add_subsystem('Inputcomp', comp, promotes=['*'])
  prob.model.add_subsystem(
      'Statecomp_Implicit',
      PropellantMassRK4Comp(num_times=n, step_size=h),
      promotes=['*'],
  )
  prob.setup()
  prob.check_partials(compact_print=True)
  
  
  # pytest
  def test_fn():
      from openmdao.utils.assert_utils import assert_check_partials
      assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)
  
::

  -----------------------------------------------------
  Component: PropellantMassRK4Comp 'Statecomp_Implicit'
  -----------------------------------------------------
  '<output>'        wrt '<variable>'              | fwd mag.   | rev mag.   | check mag. | a(fwd-chk) | a(rev-chk) | a(fwd-rev) | r(fwd-chk) | r(rev-chk) | r(fwd-rev)
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  'propellant_mass' wrt 'initial_propellant_mass' | 4.4721e+00 | 4.4721e+00 | 4.4721e+00 | 2.3047e-05 | 2.3047e-05 | 0.0000e+00 | 5.1534e-06 | 5.1534e-06 | 0.0000e+00 >ABS_TOL >REL_TOL
  'propellant_mass' wrt 'mass_flow_rate'          | 8.2704e+04 | 8.2704e+04 | 8.2704e+04 | 7.2487e-05 | 7.2487e-05 | 0.0000e+00 | 8.7646e-10 | 8.7646e-10 | 0.0000e+00 >ABS_TOL
  
  ####################################################################################
  Sub Jacobian with Largest Relative Error: PropellantMassRK4Comp 'Statecomp_Implicit'
  ####################################################################################
  '<output>'        wrt '<variable>'              | fwd mag.   | rev mag.   | check mag. | a(fwd-chk) | a(rev-chk) | a(fwd-rev) | r(fwd-chk) | r(rev-chk) | r(fwd-rev)
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------
  'propellant_mass' wrt 'initial_propellant_mass' | 4.4721e+00 | 4.4721e+00 | 4.4721e+00 | 2.3047e-05 | 2.3047e-05 | 0.0000e+00 | 5.1534e-06 | 5.1534e-06 | 0.0000e+00
  
