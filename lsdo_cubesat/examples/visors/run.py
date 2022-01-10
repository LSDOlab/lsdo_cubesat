import numpy as np
import time
import matplotlib.pyplot as plt

from lsdo_cubesat.parameters.swarm import SwarmParams
from lsdo_cubesat.parameters.cubesat import CubesatParams
from lsdo_cubesat.examples.visors.swarm_group import Swarm

from csdl_om import Simulator

from lsdo_cubesat.examples.visors.initial_plots import plot_initial
from lsdo_cubesat.examples.visors.save_data import save_data
from lsdo_cubesat.examples.visors.generate_reference_orbit import generate_reference_orbit


def make_swarm(swarm):

    # initial state relative to reference orbit
    initial_orbit_state = np.array([1e-3] * 3 + [1e-3] * 3)

    np.random.seed(0)

    cubesats = dict()

    cubesats['optics'] = CubesatParams(
        name='optics',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6) * 1e-3,
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.,
        apogee_altitude=500.,
    )

    cubesats['detector'] = CubesatParams(
        name='detector',
        dry_mass=1.3,
        initial_orbit_state=initial_orbit_state * np.random.rand(6) * 1e-3,
        approx_altitude_km=500.,
        specific_impulse=47.,
        perigee_altitude=500.002,
        apogee_altitude=499.98,
    )

    for v in cubesats.values():
        swarm.add(v)
    m = Swarm(swarm=swarm)
    start_compile = time.time()
    sim = Simulator(m)
    end_compile = time.time()
    total_compile_time = end_compile - start_compile
    print('========== TOTAL COMPILE TIME ==========')
    print(total_compile_time, 's')
    print('=========================================')
    return sim


# if True:
if False:
    num_times = 1501
    num_cp = 30
# elif False:
elif True:
    num_times = 40
    num_cp = 5
else:
    num_times = 4
    num_cp = 3

step_size = 95 * 60 / (num_times - 1)

plot_reference_orbit = True
plot_initial_outputs = True
optimize = False
ref_orbit, ax = generate_reference_orbit(
    num_times,
    step_size,
    plot=plot_reference_orbit,
)
sim = make_swarm(
    SwarmParams(
        num_times=num_times,
        num_cp=num_cp,
        step_size=step_size,
        cross_threshold=0.882,
        # cross_threshold=0.857,
        # cross_threshold=0.2,
        # cross_threshold=-0.9,
        # cross_threshold=0.9,
    ))
sim['reference_orbit_state_km'] = ref_orbit
if optimize is False:
    sim.run()
    # sim.visualize_implementation()
    if plot_reference_orbit is True:
        # plot sun direction
        sd = sim['sun_direction']
        ax.plot(
            sd[0, :] * 6000,
            sd[1, :] * 6000,
            sd[2, :] * 6000,
            'o',
        )
        plt.show()

    if plot_initial_outputs is True:
        plot_initial(sim)
else:
    from optimize import snopta, SNOPT_options
    from optimization import get_constraint_bounds, get_dv_bounds, get_problem_dimensions, get_names

    def sntoya_objF(status, x, needF, F, needG, G):
        sim.update_design_variables(x)
        sim.run()
        F[0] = sim.objective()
        F[1:] = sim.constraints()

        return status, F

    def sntoya_objFG(status, x, needF, F, needG, G):
        sim.update_design_variables(x)
        sim.run()
        F[0] = sim.objective()
        F[1:] = sim.constraints()
        # G[:] = sim.compute_total_derivatives().flatten()[n:]
        G[:] = sim.compute_total_derivatives().flatten()
        return status, F, G

    n, nF = get_problem_dimensions(sim)
    xnames, Fnames = get_names(n, nF)
    xlow, xupp = get_dv_bounds(sim)
    Flow, Fupp = get_constraint_bounds(sim)
    x0 = sim.design_variables()

    A = np.zeros((n, nF))
    G = 2 * np.ones((nF, n))
    # G[0, :] = 0
    A = None
    G = None

    options = SNOPT_options()
    options.setOption('Verbose', False)
    options.setOption('Solution print', True)
    options.setOption('Print filename', 'sntoya.out')
    options.setOption('Summary frequency', 1)

    result = snopta(sntoya_objFG,
                    n,
                    nF,
                    x0=sim.design_variables(),
                    name='sntoyaFG',
                    xlow=xlow,
                    xupp=xupp,
                    Flow=Flow,
                    Fupp=Fupp,
                    ObjRow=1,
                    A=A,
                    G=G,
                    xnames=xnames,
                    Fnames=Fnames)
    print(result)
