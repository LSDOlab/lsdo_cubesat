import numpy as np

from smt.surrogate_models import RMTB


def smt_exposure(nt, az, el, yt):
    az = np.sign(az) * np.mod(az, np.pi)
    el = np.sign(el) * np.mod(el, np.pi / 2)
    xt = np.concatenate(
        (
            az.reshape(len(az), 1),
            el.reshape(len(el), 1),
        ),
        axis=1,
    )
    xlimits = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])

    sm = RMTB(
        xlimits=xlimits,
        order=4,
        num_ctrl_pts=20,
        energy_weight=1e-3,
        regularization_weight=1e-7,
        print_global=False,
    )

    sm.set_training_values(xt, yt)
    sm.train()
    return sm
