Spacecraft Attitude Dynamics
============================

Partial derivatives
-------------------

The attitude model provies partial derivatives that OpenMDAO can use for
gradient-based optimization.
The following script checks the accuracy of the partial derivatives for
the attitude module.

.. jupyter-execute::
  ../../../lsdo_cubesat/tests/test_sc_attitude.py
