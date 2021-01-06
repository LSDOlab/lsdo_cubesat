from setuptools import find_packages, setup

setup(
    name='lsdo_cubesat',
    version='0.0.1.dev0',
    description='Large-scale optimization of CubeSat swarms',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'dash==1.2.0',
        'dash-daq==0.1.0',
        'openmdao',
        'smt',
        'pint',
        'sphinx-rtd-theme',
        'sphinx-code-include',
        'jupyter-sphinx',
        'numpydoc',
    ],
)
