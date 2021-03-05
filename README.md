# LSDO CubeSat

The LSDO CubeSat Toolbox is an open-source tool for large-scale design
optimization of CubeSats and CubeSat swarms.

It is built on top of the [OpenMDAO framework](https://openmdao.org/).

The source code is hosted on
[GitHub](https://github.com/lsdolab/lsdo_cubesat).

## To Do

- run
  - [ ] Make an Earth model to be used by communication
  - SwarmGroup (rename to VISORSGroup and move to VISORS repo)
    - ReferenceOrbitGroup
      - [ ] InitialOrbitComp
      - [ ] reference_orbit_state_km  (units)
      - [ ] radius_km?
    - [ ] AlignmentGroup (rename and move?)
      - convert to omtools
      - units
    - [ ] CubesatGroup
      - [ ] select attitude group
      - [ ] ~~SolarIlluminationComp~~
        - surogate model
      - [ ] PropulsionGroup
      - [ ] AerodynamicsGroup
      - orbit_avionics
        - [x] OrbitGroup
          - [ ] ~~RelativeOrbitRK4Comp~~
            - integrator
          - [ ] RotMtxTIComp
            - make new comp for omtools
      - [x] CommGroup
      - [ ] download rates and connections
      - [ ] BsplineComp
        - make bspline fn in omtools
      - [ ] ~~DataDownloadComp~~
        - integrator
