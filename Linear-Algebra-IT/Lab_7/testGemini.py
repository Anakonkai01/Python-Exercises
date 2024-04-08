import numpy as np

# Define the vectors
v1 = np.array([1, 0, 2])
v2 = np.array([0, 1, 4])
v3 = np.array([2, -2, -4])

# Stack the vectors as columns of a matrix
vectors = np.vstack([v1, v2, v3])

# Check the rank of the matrix (dimension of the span)
rank = np.linalg.matrix_rank(vectors)
print("Dimension of the span:", rank)

# Interpretation of rank
if rank < vectors.shape[1]:  # Less than the number of vectors
  print("The vectors are linearly dependent. The dimension of the span is less than the number of vectors.")
else:  # Equal to the number of vectors
  print("The vectors may be linearly independent (a basis).")

# Find a basis using QR decomposition (optional)
Q, R = np.linalg.qr(vectors)
basis = Q[:, :rank]  # Extract basis vectors from the first rank columns of Q

# Print the basis vectors, emphasizing non-uniqueness
print("\nOne possible basis for the span (not unique):")  # Indicate non-uniqueness
for i, vec in enumerate(basis):
  print(f"v{i+1}:", vec)
