import numpy as np

# TODO: move everything below this line to modopt
inf = 1.0e20


def get_problem_dimensions(sim):
    dv = sim.design_variables()
    return (len(dv), 1 + len(sim.constraints()))


def get_dv_bounds(sim):
    dv_meta = sim.get_design_variable_metadata()
    l = []
    u = []
    for key in sim.dv_keys:
        shape = sim[key].shape
        ll = dv_meta[key]['lower']
        uu = dv_meta[key]['upper']
        l = np.concatenate((l, (ll * np.ones(shape)).flatten()))
        u = np.concatenate((u, (uu * np.ones(shape)).flatten()))
    return (l, u)


# TODO: extract equality constraint bounds
def get_constraint_bounds(sim):
    c_meta = sim.get_constraints_metadata()
    l = [-inf]
    u = [inf]
    for key in sim.constraint_keys:
        shape = sim[key].shape
        ll = c_meta[key]['lower']
        uu = c_meta[key]['upper']
        l = np.concatenate((l, (ll * np.ones(shape)).flatten()))
        u = np.concatenate((u, (uu * np.ones(shape)).flatten()))
    return (l, u)


def get_names(n, nF):
    # Name arrays have to be dtype='|S1' and also have to be the
    # correct length, else they are ignored by SNOPT:
    xnames = np.empty(n, dtype='|S8')
    for i in range(n):
        xnames[i] = "      x{}".format(i)

    Fnames = np.empty(nF, dtype='|S8')
    for i in range(nF):
        Fnames[i] = "      F{}".format(i)

    return (xnames, Fnames)
