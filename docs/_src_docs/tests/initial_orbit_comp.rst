Initial Orbit
=============

.. code-block:: python

  from openmdao.api import Problem, Group, IndepVarComp
  from lsdo_cubesat.orbit.initial_orbit_comp import InitialOrbitComp
  import numpy as np
  
  np.random.seed(0)
  
  group = Group()
  
  comp = IndepVarComp()
  
  perigee_altitude = np.random.rand(1)
  apogee_altitude = np.random.rand(1)
  RAAN = np.random.rand(1)
  inclination = np.random.rand(1)
  argument_of_periapsis = np.random.rand(1)
  true_anomaly = np.random.rand(1)
  
  comp.add_output('perigee_altitude', val=perigee_altitude)
  comp.add_output('apogee_altitude', val=apogee_altitude)
  comp.add_output('RAAN', val=RAAN)
  comp.add_output('inclination', val=inclination)
  comp.add_output('argument_of_periapsis', val=argument_of_periapsis)
  comp.add_output('true_anomaly', val=true_anomaly)
  
  group.add_subsystem('Inputcomp', comp, promotes=['*'])
  
  group.add_subsystem('Statecomp_Implicit', InitialOrbitComp(), promotes=['*'])
  
  prob = Problem()
  prob.model = group
  prob.setup(check=True)
  prob.run_model()
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
  ------------------------------------------------
  Component: InitialOrbitComp 'Statecomp_Implicit'
  ------------------------------------------------
  '<output>'               wrt '<variable>'            | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  --------------------------------------------------------------------------------------------------------
  
  'initial_orbit_state_km' wrt 'RAAN'                  | 1.1133e+02 | 1.1133e+02 | 2.1442e-07 | 1.9260e-09
  'initial_orbit_state_km' wrt 'apogee_altitude'       | 3.1148e-04 | 3.1148e-04 | 8.7176e-08 | 2.7988e-04 >REL_TOL
  'initial_orbit_state_km' wrt 'argument_of_periapsis' | 1.1133e+02 | 1.1133e+02 | 1.2867e-07 | 1.1557e-09
  'initial_orbit_state_km' wrt 'inclination'           | 2.0826e+00 | 2.0826e+00 | 1.2851e-06 | 6.1705e-07 >ABS_TOL
  'initial_orbit_state_km' wrt 'perigee_altitude'      | 9.9997e-01 | 9.9997e-01 | 1.2909e-06 | 1.2910e-06 >ABS_TOL >REL_TOL
  'initial_orbit_state_km' wrt 'true_anomaly'          | 1.1133e+02 | 1.1133e+02 | 1.0267e-06 | 9.2220e-09 >ABS_TOL
  
  ###############################################################################
  Sub Jacobian with Largest Relative Error: InitialOrbitComp 'Statecomp_Implicit'
  ###############################################################################
  '<output>'               wrt '<variable>'            | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  --------------------------------------------------------------------------------------------------------
  'initial_orbit_state_km' wrt 'apogee_altitude'       | 3.1148e-04 | 3.1148e-04 | 8.7176e-08 | 2.7988e-04
  
