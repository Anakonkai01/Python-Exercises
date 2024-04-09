import numpy as np
from scipy.linalg import hilbert, pascal, null_space

# Create the Hilbert matrix with size 5
hilbert_matrix = hilbert(5)
# Find the null space basis for Hilbert matrix
null_basis_hilbert = null_space(hilbert_matrix)

# Create the Pascal matrix with size 5
pascal_matrix = pascal(5)
# Find the null space basis for Pascal matrix
null_basis_pascal = null_space(pascal_matrix)

# Generate a magic square with size 5
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
# Find the null space basis for Magic matrix
null_basis_magic = null_space(magic_matrix)

# Print the null space basis for each matrix
print("Basis for the null space of Hilbert matrix:")
print(null_basis_hilbert)
print("\nBasis for the null space of Pascal matrix:")
print(null_basis_pascal)
print("\nBasis for the null space of Magic matrix:")
print(null_basis_magic)
