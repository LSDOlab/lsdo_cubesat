#from distutils.core import setup
from setuptools import setup,find_packages


setup(
    name='lsdo_cubesat',
    version='1',
    packages=[
        'lsdo_cubesat',
        'lsdo_cubesat.aerodynamics',
        'lsdo_cubesat.alignment',
        'lsdo_cubesat.attitude',
        'lsdo_cubesat.orbit',
        'lsdo_cubesat.propulsion',
        'lsdo_cubesat.swarm',
        'lsdo_cubesat.utils',
        'lsdo_cubesat.viz',
        'lsdo_cubesat.newcomm',
    ],
#    packages=find_packages(),
    install_requires=[
        'dash==1.2.0',
        'dash-daq==0.1.0',
        # 'sphinx_auto_embed',
    ],
)
