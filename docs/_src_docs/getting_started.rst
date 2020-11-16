Getting Started
===============

Installation
------------

To install ``lsdo_cubesat``, first clone the Git repository from GitHub,
and then install using ``pip``.

.. code:: shell

    git clone https://github.com/lsdolab/lsdo_cubesat
    cd lsdo_cubesat
    pip install -e .

Testing
-------

Testing is performed for the ``master`` branch automatically.
To perform tests locally, first install ``pytest``, and then run
``pytest``

.. code:: shell

    pip install pytest
    cd /path/to/lsdo_cubesat/
    pytest
