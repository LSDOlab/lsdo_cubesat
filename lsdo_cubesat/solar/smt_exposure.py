import numpy as np

from lsdo_utils.miscellaneous_functions.structure_data import structure_data
from smt.surrogate_models import RMTB
# from smt.surrogate_models import RMTC


def smt_exposure(nt, az, el, yt):
    xt = np.concatenate(
        (
            az.reshape(len(az), 1),
            el.reshape(len(el), 1),
        ),
        axis=1,
    )
    xlimits = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])

    # DO NOT USE
    # sm = RMTC(
    #     xlimits=xlimits,
    #     num_elements=nt,
    #     energy_weight=1e-15,
    #     regularization_weight=0.0,
    #     print_global=False,
    # )

    sm = RMTB(
        xlimits=xlimits,
        order=4,
        num_ctrl_pts=20,
        energy_weight=1e-7,
        regularization_weight=1e-7,
        print_global=False,
    )

    sm.set_training_values(xt, yt)
    print('training...')
    sm.train()
    print('training complete')
    return sm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 10

    # load training data
    az = np.genfromtxt('lsdo_cubesat/data/arrow_xData.csv', delimiter=',')
    el = np.genfromtxt('lsdo_cubesat/data/arrow_yData.csv', delimiter=',')
    yt = np.genfromtxt('lsdo_cubesat/data/arrow_zData.csv', delimiter=',')
    step = 4
    print(az.shape)  # (400,)
    # print(
    #     az.reshape((20, 20))[::step, ::step].reshape(
    #         (int(400 / step**2 / 2), 2)))
    # print(az.reshape((20, 20))[::step, ::step])
    # print(el.reshape((20, 20))[::step, ::step])

    fig, ax = plt.subplots(1, 2)
    ax[0].contourf(az.reshape((20, 20)), el.reshape((20, 20)),
                   yt.reshape((20, 20)))
    ax[0].set_title('training data')

    # generate surrogate model
    step = 2
    sm = smt_exposure(
        int(len(yt) / step),
        az[::step],
        el[::step],
        yt[::step],
    )

    # generate predictions
    n = 800
    az = np.linspace(-np.pi, np.pi, n)
    el = np.linspace(-np.pi, np.pi, n)
    x, y = np.meshgrid(az, el, indexing='xy')
    sunlit_area = np.zeros(n**2).reshape((n, n))
    sunlit_area = sm.predict_values(
        np.concatenate(
            (
                x.reshape(n**2, 1),
                y.reshape(n**2, 1),
            ),
            axis=1,
        ))
    step = 2
    ax[1].contourf(x.reshape((n, n)), y.reshape((n, n)),
                   sunlit_area.reshape((n, n)))
    ax[1].set_title('prediction')
    plt.show()
