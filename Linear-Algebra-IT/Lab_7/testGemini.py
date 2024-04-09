import numpy as np
from scipy.linalg import null_space

def null_space_basis(matrix):
  """
  Finds a basis for the null space of a matrix.

  Args:
      matrix: A 2D numpy array representing the matrix.

  Returns:
      A list of numpy arrays representing the basis vectors of the null space.
  """

  # Calculate the null space using scipy.linalg.null_space
  nullspace_vectors = null_space(matrix)

  # Convert null space vectors to a list (optional)
  basis = nullspace_vectors.tolist()
  return basis

# Function to generate Hilbert matrix
def hilbert_matrix(n):
  H = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      H[i, j] = 1 / (i + j + 1)
  return H

# Function to generate Pascal matrix
def pascal_matrix(n):
  P = np.zeros((n, n))
  for i in range(n):
    for j in range(min(i, n - i - 1)):
      P[i, j] = np.math.factorial(i) // (np.math.factorial(j) * np.math.factorial(i - j))
  return P

# Function to generate Magic matrix (example - 3x3 works for n=5 but not guaranteed for all sizes)
def magic_matrix(n):
  if n % 2 != 1:
    raise ValueError("Magic matrix size must be odd.")
  magic_sum = n * (n + 1) // 2
  M = np.zeros((n, n))
  i, j, num = 0, n // 2, 1
  while num <= n * n:
    M[i, j] = num
    i = (i - 1 + n) % n
    j = (j + 1) % n
    num += 1
  return M

# Example usage
# (Modify these calls to create matrices of size 5)
# A = hilbert_matrix(5)
# A = pascal_matrix(5)
A = magic_matrix(5)  # Uncomment this line if using a magic matrix

basis = null_space_basis(A)
print("Basis for the null space of A:")
for vec in basis:
  print(vec)
