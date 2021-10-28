import csdl


def compute_norm_unit_vec(vec, *, num_times):
    norm = csdl.pnorm(vec, axis=0)
    unit_vec = vec / csdl.expand(norm, (3, num_times), 'i->ji')
    return norm, unit_vec
