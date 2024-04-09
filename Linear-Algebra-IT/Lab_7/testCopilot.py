import numpy as np
from scipy.linalg import hilbert, pascal

# Create the Hilbert matrix with size 5
hilbert_matrix = hilbert(5)

# Find the null space basis using singular value decomposition (SVD)
_, _, V = np.linalg.svd(hilbert_matrix)

# The last columns of V form the null space basis
null_basis_hilbert = V.T[:, -4:]

print("Basis for the null space of Hilbert matrix:")
print(null_basis_hilbert)

# Create the Pascal matrix with size 5
pascal_matrix = pascal(5)

# Find the null space basis using singular value decomposition (SVD)
_, _, V = np.linalg.svd(pascal_matrix)

# The last columns of V form the null space basis
null_basis_pascal = V.T[:, -4:]

print("Basis for the null space of Pascal matrix:")
print(null_basis_pascal)

def generate_magic_square(n):
    magic_square = np.zeros((n, n), dtype=int)
    i, j = 0, n // 2
    num = 1

    while num <= n**2:
        magic_square[i, j] = num
        num += 1
        newi, newj = (i - 1) % n, (j + 1) % n

        if magic_square[newi, newj]:
            i += 1
        else:
            i, j = newi, newj

    return magic_square

# Create a magic square with size 5
magic_matrix = generate_magic_square(5)

# Find the null space basis using SVD
_, _, V = np.linalg.svd(magic_matrix)

# The last columns of V form the null space basis
null_basis_magic = V.T[:, -4:]

print("\nBasis for the null space of Magic matrix:")
print(null_basis_magic)
