Orbit Angular Speed
===================

.. code-block:: python

  from openmdao.api import Problem, IndepVarComp
  from lsdo_cubesat.orbit.orbit_angular_speed_group import OrbitAngularSpeedGroup
  import numpy as np
  
  np.random.seed(0)
  num_times = 100
  
  leo = np.abs(np.random.rand(3, num_times)) * 10 + 6371 + 150
  
  comp = IndepVarComp()
  comp.add_output('position_km', val=leo)
  comp.add_output('velocity_km_s', val=np.random.rand(3, num_times))
  
  prob = Problem()
  prob.model.add_subsystem(
      'indeps',
      comp,
      promotes=['*'],
  )
  prob.model.add_subsystem(
      'orbit_angular_speed_group',
      OrbitAngularSpeedGroup(num_times=num_times),
      promotes=['*'],
  )
  prob.setup()
  prob.run_model()
  cp = prob.check_partials(compact_print=True)
  
  
  # pytest
  def test_fn():
      from openmdao.utils.assert_utils import assert_check_partials
      assert_check_partials(cp, atol=1.e-6, rtol=1.e-6)
  
::

  -------------------------------------------------------------------------------
  Component: CrossProductComp 'orbit_angular_speed_group.compute_sp_ang_momentum'
  -------------------------------------------------------------------------------
  '<output>'            wrt '<variable>'    | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  ---------------------------------------------------------------------------------------------
  
  'sp_ang_momentum_vec' wrt 'position_km'   | 1.4085e+01 | 1.4085e+01 | 9.0495e-06 | 6.4250e-07 >ABS_TOL
  'sp_ang_momentum_vec' wrt 'velocity_km_s' | 1.5985e+05 | 1.5985e+05 | 7.6587e-06 | 4.7910e-11 >ABS_TOL
  ------------------------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_sp_ang_momentum_mag.compute_square'
  ------------------------------------------------------------------------------------------------------
  '<output>'                    wrt '<variable>'          | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------------------------
  
  'sp_ang_momentum_vec_squared' wrt 'sp_ang_momentum_vec' | 9.4364e+04 | 9.4364e+04 | 3.0196e-02 | 3.1999e-07 >ABS_TOL
  ---------------------------------------------------------------------------------------------------
  Component: ArrayContractionComp 'orbit_angular_speed_group.compute_sp_ang_momentum_mag.compute_sum'
  ---------------------------------------------------------------------------------------------------
  '<output>'                        wrt '<variable>'                  | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------------------------------------
  
  'sum_sp_ang_momentum_vec_squared' wrt 'sp_ang_momentum_vec_squared' | 1.7321e+01 | 1.7322e+01 | 3.7039e-02 | 2.1383e-03 >ABS_TOL >REL_TOL
  -----------------------------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_sp_ang_momentum_mag.compute_square_root'
  -----------------------------------------------------------------------------------------------------------
  '<output>'            wrt '<variable>'                      | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  ---------------------------------------------------------------------------------------------------------------
  
  'sp_ang_momentum_mag' wrt 'sum_sp_ang_momentum_vec_squared' | 2.1494e-03 | 2.1492e-03 | 3.1074e-06 | 1.4458e-03 >ABS_TOL >REL_TOL
  -------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_semi_latus_rectum'
  -------------------------------------------------------------------------------------
  '<output>'          wrt '<variable>'          | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -------------------------------------------------------------------------------------------------
  
  'semi_latus_rectum' wrt 'mu'                  | 1.6977e-03 | 1.6977e-03 | 6.3292e-08 | 3.7282e-05 >REL_TOL
  'semi_latus_rectum' wrt 'sp_ang_momentum_mag' | 2.3674e-01 | 2.3674e-01 | 1.0637e-07 | 4.4933e-07
  -----------------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_position_mag.compute_square'
  -----------------------------------------------------------------------------------------------
  '<output>'            wrt '<variable>'  | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -------------------------------------------------------------------------------------------
  
  'position_km_squared' wrt 'position_km' | 2.2607e+05 | 2.2607e+05 | 9.4841e-02 | 4.1952e-07 >ABS_TOL
  --------------------------------------------------------------------------------------------
  Component: ArrayContractionComp 'orbit_angular_speed_group.compute_position_mag.compute_sum'
  --------------------------------------------------------------------------------------------
  '<output>'                wrt '<variable>'          | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -------------------------------------------------------------------------------------------------------
  
  'sum_position_km_squared' wrt 'position_km_squared' | 1.7321e+01 | 1.7274e+01 | 2.0838e-01 | 1.2064e-02 >ABS_TOL >REL_TOL
  ----------------------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_position_mag.compute_square_root'
  ----------------------------------------------------------------------------------------------------
  '<output>'     wrt '<variable>'              | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  ------------------------------------------------------------------------------------------------
  
  'position_mag' wrt 'sum_position_km_squared' | 4.4234e-04 | 4.3970e-04 | 7.3422e-06 | 1.6698e-02 >ABS_TOL >REL_TOL
  -----------------------------------------------------------------------------
  Component: ArrayExpansionComp 'orbit_angular_speed_group.expand_position_mag'
  -----------------------------------------------------------------------------
  '<output>'         wrt '<variable>'   | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------
  
  'position_mag_3xn' wrt 'position_mag' | 1.7321e+01 | 1.7321e+01 | 5.8636e-06 | 3.3854e-07 >ABS_TOL
  ------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.normalize_position'
  ------------------------------------------------------------------------------
  '<output>'             wrt '<variable>'       | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -------------------------------------------------------------------------------------------------
  
  'position_unit_vector' wrt 'position_km'      | 1.5323e-03 | 1.5323e-03 | 9.8202e-10 | 6.4087e-07
  'position_unit_vector' wrt 'position_mag_3xn' | 8.8469e-04 | 8.8469e-04 | 1.0903e-09 | 1.2324e-06 >REL_TOL
  -------------------------------------------------------------------------
  Component: CrossProductComp 'orbit_angular_speed_group.compute_v_cross_h'
  -------------------------------------------------------------------------
  '<output>'                  wrt '<variable>'          | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  ---------------------------------------------------------------------------------------------------------
  
  'vel_cross_sp_ang_momentum' wrt 'sp_ang_momentum_vec' | 1.4085e+01 | 1.4085e+01 | 5.9361e-06 | 4.2145e-07 >ABS_TOL
  'vel_cross_sp_ang_momentum' wrt 'velocity_km_s'       | 6.6725e+04 | 6.6725e+04 | 5.8357e-06 | 8.7458e-11 >ABS_TOL
  -------------------------------------------------------------------
  Component: ArrayExpansionComp 'orbit_angular_speed_group.expand_mu'
  -------------------------------------------------------------------
  '<output>' wrt '<variable>' | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -------------------------------------------------------------------------------
  
  'mu_3xn'   wrt 'mu'         | 1.7321e+01 | 1.7321e+01 | 1.3189e-04 | 7.6144e-06 >ABS_TOL >REL_TOL
  ---------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_v_cross_h__mu'
  ---------------------------------------------------------------------------------
  '<output>'      wrt '<variable>'                | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  ---------------------------------------------------------------------------------------------------
  
  'v_cross_h__mu' wrt 'mu_3xn'                    | 3.0421e-07 | 3.0420e-07 | 9.1556e-12 | 3.0097e-05 >REL_TOL
  'v_cross_h__mu' wrt 'vel_cross_sp_ang_momentum' | 4.3453e-05 | 4.3453e-05 | 1.0150e-11 | 2.3358e-07
  ----------------------------------------------------------------------------------------
  Component: LinearCombinationComp 'orbit_angular_speed_group.compute_eccentricity_vector'
  ----------------------------------------------------------------------------------------
  '<output>'         wrt '<variable>'           | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -------------------------------------------------------------------------------------------------
  
  'eccentricity_vec' wrt 'position_unit_vector' | 1.7321e+01 | 1.7321e+01 | 4.9806e-10 | 2.8756e-11
  'eccentricity_vec' wrt 'v_cross_h__mu'        | 1.7321e+01 | 1.7321e+01 | 1.5056e-09 | 8.6927e-11
  -----------------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_eccentricity.compute_square'
  -----------------------------------------------------------------------------------------------
  '<output>'                 wrt '<variable>'       | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------------------
  
  'eccentricity_vec_squared' wrt 'eccentricity_vec' | 1.4741e+01 | 1.4742e+01 | 1.7320e-05 | 1.1749e-06 >ABS_TOL >REL_TOL
  --------------------------------------------------------------------------------------------
  Component: ArrayContractionComp 'orbit_angular_speed_group.compute_eccentricity.compute_sum'
  --------------------------------------------------------------------------------------------
  '<output>'                     wrt '<variable>'               | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------------------------------
  
  'sum_eccentricity_vec_squared' wrt 'eccentricity_vec_squared' | 1.7321e+01 | 1.7321e+01 | 9.0759e-10 | 5.2400e-11
  ----------------------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_eccentricity.compute_square_root'
  ----------------------------------------------------------------------------------------------------
  '<output>'     wrt '<variable>'                   | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------------------
  
  'eccentricity' wrt 'sum_eccentricity_vec_squared' | 6.7839e+00 | 6.7839e+00 | 3.1221e-06 | 4.6023e-07 >ABS_TOL
  ----------------------------------------------------------------------------------------------------
  Component: LinearPowerCombinationComp 'orbit_angular_speed_group.compute_semimajor_axis_denominator'
  ----------------------------------------------------------------------------------------------------
  '<output>'                   wrt '<variable>'   | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  ---------------------------------------------------------------------------------------------------
  
  'semimajor_axis_denominator' wrt 'eccentricity' | 1.4741e+01 | 1.4742e+01 | 1.0000e-05 | 6.7839e-07 >ABS_TOL
  ----------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_semimajor_axis'
  ----------------------------------------------------------------------------------
  '<output>'       wrt '<variable>'                 | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------------------
  
  'semimajor_axis' wrt 'semi_latus_rectum'          | 2.1899e+01 | 2.1899e+01 | 1.1935e-07 | 5.4501e-09
  'semimajor_axis' wrt 'semimajor_axis_denominator' | 3.3308e+03 | 3.3308e+03 | 7.3924e-03 | 2.2194e-06 >ABS_TOL >REL_TOL
  ------------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_orbit_angular_speed_sq'
  ------------------------------------------------------------------------------------------
  '<output>'               wrt '<variable>'     | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -------------------------------------------------------------------------------------------------
  
  'orbit_angular_speed_sq' wrt 'mu'             | 1.2792e-01 | 1.2792e-01 | 2.7357e-06 | 2.1385e-05 >ABS_TOL >REL_TOL
  'orbit_angular_speed_sq' wrt 'semimajor_axis' | 6.5038e+04 | 6.5038e+04 | 5.6246e-02 | 8.6482e-07 >ABS_TOL
  --------------------------------------------------------------------------------------------------
  Component: PowerCombinationComp 'orbit_angular_speed_group.compute_osculating_orbit_angular_speed'
  --------------------------------------------------------------------------------------------------
  '<output>'                       wrt '<variable>'             | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------------------------------
  
  'osculating_orbit_angular_speed' wrt 'orbit_angular_speed_sq' | 1.7813e+01 | 1.7813e+01 | 3.6136e-04 | 2.0287e-05 >ABS_TOL >REL_TOL
  
  ###################################################################################################################################
  Sub Jacobian with Largest Relative Error: PowerCombinationComp 'orbit_angular_speed_group.compute_position_mag.compute_square_root'
  ###################################################################################################################################
  '<output>'                       wrt '<variable>'             | fwd mag.   | check mag. | a(fwd-chk) | r(fwd-chk)
  -----------------------------------------------------------------------------------------------------------------
  'position_mag' wrt 'sum_position_km_squared' | 4.4234e-04 | 4.3970e-04 | 7.3422e-06 | 1.6698e-02
  
