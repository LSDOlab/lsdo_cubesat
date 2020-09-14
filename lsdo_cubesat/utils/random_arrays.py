import numpy as np


def make_random_bounded_array(n, bound):
    rba = np.random.rand(n)
    b1 = (rba > bound).astype(int)
    b2 = (rba < -bound).astype(int)
    rba *= b1 * b2
    return rba


def make_random_signed_array(n, sgn=1, bound=0):
    rpa = np.random.rand(n)
    b1 = (rpa > 0).astype(int)
    b2 = (rpa < 0).astype(int)

    # ensure sgn has magnitude 1
    if sgn < 0:
        sgn = -1
    else:
        sgn = 1

    # make positve array
    rpa = b1 * rpa - b2 * rpa

    # ensure bound is positive
    if bound < 0:
        bound = -bound

    if bound > 0:
        b3 = (rpa >= bound).astype(int)
        while np.any(b3 > 0):
            rpa -= b3 * bound
            b3 = (rpa >= bound).astype(int)
    return sgn * rpa


a = make_random_signed_array(10, sgn=-1, bound=0.5)
print(a)
