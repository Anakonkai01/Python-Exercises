import numpy as np

# Define the vectors
v1 = np.array([1, 0, 2])
v2 = np.array([0, 1, 4])
v3 = np.array([2, -2, -4])

# Create a matrix with the vectors as columns
A = np.column_stack((v1, v2, v3))

# Perform Reduced Row Echelon Form (RREF) on the matrix
U, s, V = np.linalg.svd(A, full_matrices=False)
rank = np.sum(s > 1e-10)

# Basis for the span
basis = U[:, :rank]

# Print the dimension of the span and the basis
print(f"The dimension of the span is: {rank}")
print("A basis for their span is:")
print(basis)
