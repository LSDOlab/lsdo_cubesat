import numpy as np
A = np.array([[1,2,3],[2,2,2],[3,6,9]])
B = np.einsum('ij->ji',A)
print(A)
print(B)
