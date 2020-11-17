Orbit Disciplines
=================

Initial Orbit
-------------

Use this component to model the rotation of a rigid body. This component
does not model moments that depend on orientation (e.g. moments due to
gravity gradient, drag).

.. autoclass:: lsdo_cubesat.orbit.initial_orbit_comp.InitialOrbitComp

Orbit Angular Speed
-------------------

.. autoclass:: lsdo_cubesat.orbit.orbit_angular_speed_group.OrbitAngularSpeedGroup

State Decomposition
-------------------

.. autoclass:: lsdo_cubesat.orbit.orbit_state_decomposition_comp.OrbitStateDecompositionComp

Reference Orbit Integrator Component (for Swarms)
-------------------------------------------------

.. autoclass:: lsdo_cubesat.orbit.reference_orbit_rk4_comp.ReferenceOrbitRK4Comp

Reference Orbit Group (for Swarms)
----------------------------------

.. autoclass:: lsdo_cubesat.orbit.reference_orbit_group.ReferenceOrbitGroup
