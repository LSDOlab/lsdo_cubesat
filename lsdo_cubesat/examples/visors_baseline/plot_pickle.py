import pickle
from lsdo_cubesat.examples.visors_baseline.plot_sim_state import plot_sim_state
with open('filename.pkl', 'rb') as handle:
    loaded_data = pickle.load(handle)
plot_sim_state(loaded_data)
