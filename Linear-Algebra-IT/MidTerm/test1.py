import numpy as np
import sympy as sp


rows = 10
cols = 10
a_matrix = np.arange(1, rows * cols + 1 ) # Create a 1D array with sequential numbers
a_matrix = a_matrix.reshape((rows, cols))


for i in a_matrix:
    for j in i:
        print(j," ",end="")
    print()