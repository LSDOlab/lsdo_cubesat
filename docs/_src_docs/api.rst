API
===

The disciplines provided by LSDO CubeSat are split into OpenMDAO
Component classes and Group classes.
Component classes model simple disciplines while Group classes package
disciplines together, simplifying model construction.

In addition to Component and Group classes for modeling the CubeSat
disciplines, LSDO CubeSat provides ``OptionsDictionary`` classes to
initialize each discipline.

Finally, LSDO CubeSat provides functions to generate training data for
surrogate models.

.. toctree::
  :maxdepth: 2
  :titlesonly:

  api/options
  api/groups
  api/components
  api/training
