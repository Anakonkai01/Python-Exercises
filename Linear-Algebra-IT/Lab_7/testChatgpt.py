import numpy as np

# Define the vectors as a matrix
vectors = np.array([[1, 0, 2], [0, 1, 4], [2, -2, -4]])

# Compute the rank of the matrix formed by the vectors
rank = np.linalg.matrix_rank(vectors)

# If the rank is equal to the number of vectors, they are linearly independent
if rank == len(vectors):
    print("The vectors are linearly independent.")
    print("The dimension of their span is:", len(vectors))
    print("A basis for their span is the set of all vectors.")
else:
    print("The vectors are linearly dependent.")
    print("The dimension of their span is less than the number of vectors.")

    # To find a basis for their span, we can use the row echelon form (rref) of the matrix
    rref, pivot_columns = np.linalg.qr(vectors.T, mode='complete')

    # Convert pivot_columns to integer indices
    pivot_indices = np.where(np.abs(np.diag(rref)) > 1e-10)[0]

    # Extract linearly independent vectors from the original matrix using the pivot indices
    basis_vectors = vectors[:, pivot_indices]

    print("A basis for their span is:")
    for vector in basis_vectors.T:
        print(vector)
