import numpy as np
fpath = os.path.dirname(os.path.realpath(__file__))
rawG_file = fpath + '/data/Comm/Gain.txt'

rawGdata = np.genfromtxt(rawG_file)
rawG = (10 ** (rawGdata / 10.0)).reshape((361, 361), order='F')
