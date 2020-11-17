Spacecraft Attitude Dynamics
============================

Partial derivatives
-------------------

The attitude model provies partial derivatives that OpenMDAO can use for
gradient-based optimization.
The following script checks the accuracy of the partial derivatives for
the attitude module.

.. code-block:: python

  from openmdao.api import Problem, Group
  from openmdao.api import IndepVarComp
  from lsdo_cubesat.utils.random_arrays import make_random_bounded_array
  from lsdo_cubesat.attitude.attitude_rk4_gravity_comp import AttitudeRK4GravityComp
  import numpy as np
  
  np.random.seed(0)
  num_times = 100
  step_size = 1e-8
  I = np.array([90, 100, 80])
  wq0 = np.random.rand(7) - 0.5
  wq0[3:] /= np.linalg.norm(wq0[3:])
  
  comp = IndepVarComp()
  comp.add_output('initial_angular_velocity_orientation', val=wq0)
  comp.add_output(
      'external_torques_x',
      val=make_random_bounded_array(num_times, bound=1).reshape((1, num_times)),
      shape=(1, num_times),
  )
  comp.add_output(
      'external_torques_y',
      val=make_random_bounded_array(num_times, bound=1).reshape((1, num_times)),
      shape=(1, num_times),
  )
  comp.add_output(
      'external_torques_z',
      val=make_random_bounded_array(num_times, bound=1).reshape((1, num_times)),
      shape=(1, num_times),
  )
  prob = Problem()
  prob.model.add_subsystem('inputs_comp', comp, promotes=['*'])
  prob.model.add_subsystem(
      'comp',
      AttitudeRK4GravityComp(
          num_times=num_times,
          step_size=step_size,
          moment_inertia_ratios=np.array([2.0 / 3.0, -2.0 / 3.0, 0]),
      ),
      promotes=['*'],
  )
  
  prob.setup(check=True, force_alloc_complex=True)
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
  ----------------------------------------
  Component: AttitudeRK4GravityComp 'comp'
  ----------------------------------------
  '<output>'                     wrt '<variable>'                           | fwd mag.   | rev mag.   | check mag. | a(fwd-chk) | a(rev-chk) | a(fwd-rev) | r(fwd-chk) | r(rev-chk) | r(fwd-rev)
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  
  'angular_velocity_orientation' wrt 'external_torques_x'                   | 7.0356e-07 | 7.0356e-07 | 7.0317e-07 | 3.1582e-09 | 3.1582e-09 | 0.0000e+00 | 4.4914e-03 | 4.4914e-03 | 0.0000e+00 >REL_TOL
  'angular_velocity_orientation' wrt 'external_torques_y'                   | 7.0356e-07 | 7.0356e-07 | 7.0313e-07 | 4.4341e-09 | 4.4341e-09 | 0.0000e+00 | 6.3062e-03 | 6.3062e-03 | 0.0000e+00 >REL_TOL
  'angular_velocity_orientation' wrt 'external_torques_z'                   | 7.0356e-07 | 7.0356e-07 | 7.0366e-07 | 3.9656e-09 | 3.9656e-09 | 0.0000e+00 | 5.6357e-03 | 5.6357e-03 | 0.0000e+00 >REL_TOL
  'angular_velocity_orientation' wrt 'initial_angular_velocity_orientation' | 2.6458e+01 | 2.6458e+01 | 2.4495e+01 | 1.0000e+01 | 1.0000e+01 | 3.6453e-20 | 4.0825e-01 | 4.0825e-01 | 1.4882e-21 >ABS_TOL >REL_TOL
  'angular_velocity_orientation' wrt 'osculating_orbit_angular_speed'       | 1.5482e-09 | 1.5482e-09 | 3.5900e-09 | 3.4410e-09 | 3.4410e-09 | 0.0000e+00 | 9.5848e-01 | 9.5848e-01 | 0.0000e+00 >REL_TOL
  
  #######################################################################
  Sub Jacobian with Largest Relative Error: AttitudeRK4GravityComp 'comp'
  #######################################################################
  '<output>'                     wrt '<variable>'                           | fwd mag.   | rev mag.   | check mag. | a(fwd-chk) | a(rev-chk) | a(fwd-rev) | r(fwd-chk) | r(rev-chk) | r(fwd-rev)
  ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  'angular_velocity_orientation' wrt 'osculating_orbit_angular_speed'       | 1.5482e-09 | 1.5482e-09 | 3.5900e-09 | 3.4410e-09 | 3.4410e-09 | 0.0000e+00 | 9.5848e-01 | 9.5848e-01 | 0.0000e+00
  
