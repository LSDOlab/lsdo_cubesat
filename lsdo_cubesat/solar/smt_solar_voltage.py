import os

import numpy as np

from smt.surrogate_models import RMTB

nt = 20


def smt_solar_voltage(dat):
    nT, nA, nI = dat[:3]
    nT = int(nT)
    nA = int(nA)
    nI = int(nI)
    T = dat[3:3 + nT]
    A = dat[3 + nT:3 + nT + nA]
    I = dat[3 + nT + nA:3 + nT + nA + nI]
    V = dat[3 + nT + nA + nI:].reshape((nT, nA, nI), order='F')
    R = np.zeros(20 * 20 * 200).reshape((20, 20, 200))
    for i in range(20):
        for j in range(20):
            R[i, j, :] = V[i, j, :] / I
    print(R.shape)
    print(T.shape)  # (20,)
    print(A.shape)  # (20,)
    print(I.shape)  # (200,)
    print(V.shape)  # (20, 20, 200)

    print(T[0])
    print(A[0])
    print(I[0])
    print(V[:, 10, 30])

    # self.MBI = MBI(V, [T, A, I], [6, 6, 15], [3, 3, 3])

    # MBI assumes the data is structured. V contains the training
    # outputs so it is nx1 x nx2 x nx3. T, A, I are the three inputs but
    # the real training inputs array would be a tensor product of the
    # three. 6,6,15 are the number of control points, 3,3,3 are the
    # B-spline order in each direction.

    # order=(3,3,3)
    # num_ctrl_pts=(6,6,15)

    # model voltage at highest temperature
    # V = V[-1, :, :]
    # print(V.shape)  # (20, 20, 200)
    # print(I)
    # print(A)

    # FIXME: not the same length
    xt = np.concatenate(
        (
            T.reshape(len(T), 1),
            I.reshape(len(I), 1),
            A.reshape(len(A), 1),
        ),
        axis=1,
    )
    print(xt)

    # required
    xlimits = np.array([
        [min(T), max(T)],
        [min(A), max(A)],
        [min(I), max(I)],
    ], )

    sm = RMTB(
        xlimits=xlimits,
        num_elements=nt,
        energy_weight=1e-15,
        regularization_weight=0.0,
        print_global=False,
    )

    # TODO: flatten...
    sm.set_training_values(xt, V)
    sm.train()
    return sm


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    n = 10

    # load training data
    dat = np.genfromtxt('cadre_iv_curve.dat', delimiter='\n')
    sm = smt_solar_voltage(dat)

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contourf(az.reshape((20, 20)), el.reshape((20, 20)), yt.reshape(
        (20, 20)))
    plt.show()

    step = 1
    print(len(az[::step]))
    print(len(yt[::step]))
    print(int(len(yt) / step))

    # generate surrogate model
    sm = smt_solar_voltage(
        int(len(yt) / step),
        az[::step],
        el[::step],
        yt[::step],
    )
    az = np.linspace(-np.pi, np.pi, n)
    el = np.linspace(-np.pi, np.pi, n)
    x, y = np.meshgrid(az, el)
    print(x.shape)
    print(y.shape)
    print('predicting sunlit area...')
    sunlit_area = sm.predict_values(
        np.array([x.flatten(), y.flatten()]).reshape((n**2, 2)), ).reshape(
            (n, n))
    print(sunlit_area.shape)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.contourf(x, y, sunlit_area)
    plt.show()
