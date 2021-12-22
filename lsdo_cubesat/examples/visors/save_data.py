from lsdo_cubesat.dash import Dash


def split_top_level_low_level_varnames(t, l, s):
    try:
        ll = s.rindex('.')
        l.append(s)
    except:
        t.append(s)


def remove_automatically_named_variables(t, l):
    remove = []
    for s in l:
        try:
            _ = s.rindex('._')
            remove.append(s)
        except:
            pass
    for r in remove:
        l.remove(r)

    remove = []
    for s in t:
        if s[0] == '_':
            remove.append(s)
    for r in remove:
        t.remove(r)


def save_data(sim):
    # save data
    a = sim.prob.model.list_outputs(
        explicit=True,
        implicit=True,
        val=False,
        prom_name=True,
        residuals=False,
        residuals_tol=None,
        units=False,
        shape=False,
        global_shape=False,
        bounds=False,
        scaling=False,
        desc=False,
        hierarchical=False,
        print_arrays=False,
        tags=None,
        includes=None,
        excludes=None,
        all_procs=False,
        list_autoivcs=False,
        out_stream=None,
        values=None,
    )
    all_variable_names = []
    for (k, v) in a:
        all_variable_names.append(v['prom_name'])
    t = []
    l = []
    _ = [
        split_top_level_low_level_varnames(t, l, s) for s in all_variable_names
    ]
    remove_automatically_named_variables(t, l)
    varnames = t + l
    dashboard = Dash(varnames)
    sim.add_recorder(dashboard.get_recorder())
