import numpy as np

from openmdao.api import ExplicitComponent

# https://www.astro.rug.nl/software/kapteyn-beta/_downloads/attitude.pdf


def get_mtx_norms(mtx):
    norm0 = np.linalg.norm(mtx, axis=0)
    norm1 = np.linalg.norm(mtx, axis=1)
    np.einsum('ik,j->ijk', norm0, np.ones(3))
    # mtx


np.random.seed(0)
num_times = 1
shape = (3, 3, num_times)
mtx = np.random.rand(np.prod(shape)).reshape(shape)
print(mtx)
norm0 = np.linalg.norm(mtx, axis=0)
print(norm0)
norm0 = np.einsum('ik,j->ijk', norm0, np.ones(3))
print(norm0)
print(mtx / norm0)
print(np.linalg.norm(mtx, axis=0))
