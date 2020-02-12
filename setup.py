from distutils.core import setup


setup(
    name='lsdo_cubesat',
    version='1',
    packages=[
        'lsdo_cubesat',
    ],
    install_requires=[
        'dash==1.2.0',
        'dash-daq==0.1.0',
        # 'sphinx_auto_embed',
    ],
)