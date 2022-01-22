from lsdo_cubesat.examples.visors.get_all_varnames import get_all_varnames
from lsdo_cubesat.dash import Dash
from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.examples.visors.make_swarm import make_swarm

# only build sim to get variable names
sim = make_swarm(
    SwarmParams(
        num_times=10,
        duration=1.,
        num_cp=2,
        step_size=1.,
        cross_threshold=0.7,
    ))
varnames = get_all_varnames(sim)
del sim

dashboard = Dash(varnames=varnames)
dashboard.run_GUI()
