from setuptools import find_packages
from setuptools import setup


setup(
    name='lsdo_cubesat',
    version='0.0.1.dev0',
    description='VISORS CubeSats Design Optimization',
    packages=find_packages(),
    author = "LSDO_Lab",
    install_requires=[
        'dash==1.2.0',
        'dash-daq==0.1.0',
        # 'sphinx_auto_embed',
    ],
)
