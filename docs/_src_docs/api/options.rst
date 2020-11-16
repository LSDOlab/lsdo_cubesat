Options Dictionaries
====================

LSDO CubeSat provides several dictionary-like classes to simplify
setting options for Groups and Components.
These classes enable the user to create, for example, several instances
of ground stations with different lattitudes and longitudes, or several
CubeSats with different orbit inclinations.
The components provided by LSDO CubeSat read these values at model
setup.

Ground_station
--------------

.. list-table:: List of options
  :header-rows: 1
  :widths: 15, 10, 20, 20, 30
  :stub-columns: 0

  *  -  Option
     -  Default
     -  Acceptable values
     -  Acceptable types
     -  Description
  *  -  name
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['str']
     -  
  *  -  lon
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['float']
     -  
  *  -  lat
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['float']
     -  
  *  -  alt
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['float']
     -  
  *  -  antAngle
     -  2.0
     -  None
     -  ['float']
     -  

Cubesat
-------

.. list-table:: List of options
  :header-rows: 1
  :widths: 15, 10, 20, 20, 30
  :stub-columns: 0

  *  -  Option
     -  Default
     -  Acceptable values
     -  Acceptable types
     -  Description
  *  -  name
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['str']
     -  
  *  -  dry_mass
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['float']
     -  
  *  -  initial_orbit_state
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['ndarray']
     -  
  *  -  approx_altitude_km
     -  **Required**
     -  None
     -  ['float']
     -  
  *  -  acceleration_due_to_gravity
     -  9.81
     -  None
     -  ['float']
     -  
  *  -  specific_impulse
     -  47.0
     -  None
     -  ['float']
     -  
  *  -  perigee_altitude
     -  500.1
     -  None
     -  ['float']
     -  
  *  -  apogee_altitude
     -  499.9
     -  None
     -  ['float']
     -  
  *  -  RAAN
     -  66.279
     -  None
     -  ['float']
     -  
  *  -  inclination
     -  82.072
     -  None
     -  ['float']
     -  
  *  -  argument_of_periapsis
     -  0.0
     -  None
     -  ['float']
     -  
  *  -  true_anomaly
     -  337.987
     -  None
     -  ['float']
     -  
  *  -  radius_earth_km
     -  6371.0
     -  None
     -  ['float']
     -  

Swarm
-----

.. list-table:: List of options
  :header-rows: 1
  :widths: 15, 10, 20, 20, 30
  :stub-columns: 0

  *  -  Option
     -  Default
     -  Acceptable values
     -  Acceptable types
     -  Description
  *  -  num_times
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['int']
     -  
  *  -  num_cp
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['int']
     -  
  *  -  step_size
     -  <object object at 0x7ffe36753e80>
     -  None
     -  ['float']
     -  
  *  -  cross_threshold
     -  -0.87
     -  None
     -  ['float']
     -  
  *  -  launch_date
     -  0.0
     -  None
     -  ['float']
     -  
