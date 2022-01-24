import pickle
from lsdo_cubesat.examples.visors.plot_sim_state import plot_sim_state
with open('filename.pickle', 'rb') as handle:
    sim = pickle.load(handle)
plot_sim_state(sim)
