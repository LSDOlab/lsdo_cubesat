import numpy as np

from smt.surrogate_models import RMTB

# from smt.surrogate_models import RMTC


def smt_exposure(nt, az, el, yt):
    print(np.min(az))
    print(np.max(az))
    print(np.min(el))
    print(np.max(el))

    az = np.sign(az) * np.mod(az, np.pi)
    # el = np.sign(el) * np.mod(el, np.pi / 2)
    xt = np.concatenate(
        (
            az.reshape(len(az), 1),
            el.reshape(len(el), 1),
        ),
        axis=1,
    )
    xlimits = np.array([[-np.pi, np.pi], [-np.pi, np.pi]])
    # xlimits = np.array([[-np.pi, np.pi], [-np.pi / 2, np.pi / 2]])

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
        order=6,
        num_ctrl_pts=20,
        energy_weight=1e-3,
        # energy_weight=1e-4,
        regularization_weight=1e-7,
        # regularization_weight=1e-4,
        print_global=False,
    )

    sm.set_training_values(xt, yt)
    sm.train()
    return sm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 10

    # load training data
    az = np.genfromtxt('../training_data/arrow_xData.csv', delimiter=',')
    el = np.genfromtxt('../training_data/arrow_yData.csv', delimiter=',')
    yt = np.genfromtxt('../training_data/arrow_zData.csv', delimiter=',')
    step = 4
    print(az.shape)  # (400,)
    # print(
    #     az.reshape((20, 20))[::step, ::step].reshape(
    #         (int(400 / step**2 / 2), 2)))
    # print(az.reshape((20, 20))[::step, ::step])
    # print(el.reshape((20, 20))[::step, ::step])

    fig, ax = plt.subplots(1, 4)
    CS = ax[0].contourf(
        az.reshape((20, 20)),
        el.reshape((20, 20)),
        yt.reshape((20, 20)),
        cmap=plt.cm.bone,
    )
    CS.cmap.set_under('black')
    CS.cmap.set_over('white')
    # ax[0].clabel(CS, inline=1, fontsize=10)
    ax[0].clabel(CS)
    ax[0].set_title('training data')

    # generate surrogate model
    step = 1
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
    rp = np.concatenate(
        (
            x.reshape(n**2, 1),
            y.reshape(n**2, 1),
        ),
        axis=1,
    )
    sunlit_area = np.zeros(n**2).reshape((n, n))
    sunlit_area = sm.predict_values(rp)
    if np.min(sunlit_area) < 0:
        sunlit_area -= np.min(sunlit_area)
    else:
        sunlit_area += np.min(sunlit_area)
    max_sunlit_area = min(1, np.max(sunlit_area))
    sunlit_area /= np.max(sunlit_area)
    sunlit_area *= max_sunlit_area
    print(np.min(sunlit_area))
    print(np.max(sunlit_area))
    step = 2

    dadx = sm.predict_derivatives(
        rp,
        0,
    )

    dady = sm.predict_derivatives(
        rp,
        1,
    )

    CS = ax[1].contourf(
        x.reshape((n, n)),
        y.reshape((n, n)),
        sunlit_area.reshape((n, n)),
        cmap=plt.cm.bone,
    )
    ax[1].set_title('prediction')
    CS = ax[2].contourf(
        x.reshape((n, n)),
        y.reshape((n, n)),
        dadx.reshape((n, n)),
        cmap=plt.cm.bone,
    )
    ax[2].set_title('prediction (dadx)')
    CS = ax[3].contourf(
        x.reshape((n, n)),
        y.reshape((n, n)),
        dady.reshape((n, n)),
        cmap=plt.cm.bone,
    )
    ax[3].set_title('prediction (dady)')
    plt.show()
