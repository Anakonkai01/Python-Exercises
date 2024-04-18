import numpy as np
import matplotlib.pyplot as plt

def transform_vector(v, S):
  """
  This function multiplies a 2D vector v by the transformation matrix S.

  Args:
      v: A NumPy array representing the 2D vector (x, y).
      S: A 2x2 NumPy array representing the transformation matrix Sλ,µ.

  Returns:
      A NumPy array representing the transformed vector (x', y').
  """
  return np.dot(S, v)

def visualize_transformation(v, S, label):
  """
  This function visualizes the original and transformed vectors with arrows.

  Args:
      v: A NumPy array representing the original 2D vector (x, y).
      S: A 2x2 NumPy array representing the transformation matrix Sλ,µ.
      label: A string label for the transformation.
  """
  origin = np.array([0, 0])  # Origin point
  transformed_v = transform_vector(v, S)

  # Plot arrows for original and transformed vectors
  plt.arrow(origin[0], origin[1], v[0], v[1], head_width=0.1, head_length=0.1, label='Original')
  plt.arrow(origin[0], origin[1], transformed_v[0], transformed_v[1], head_width=0.1, head_length=0.1, label=label, color='red')

  # Set axis limits slightly larger than vector coordinates
  plt.xlim(-max(abs(v[0]), abs(transformed_v[0])) * 1.1, max(abs(v[0]), abs(transformed_v[0])) * 1.1)
  plt.ylim(-max(abs(v[1]), abs(transformed_v[1])) * 1.1, max(abs(v[1]), abs(transformed_v[1])) * 1.1)

  # Label axes and add title
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title(label + ' Transformation')
  plt.legend()
  plt.grid(True)
  plt.show()

# Define a sample vector
v = np.array([2, 1])  # You can change this vector for different visualizations

# Examples of transformations with visualizations
S22 = np.array([[2, 0], [0, 2]])
visualize_transformation(v, S22, 'S2,2')

S0_50_5 = np.array([[0.5, 0], [0, 0.5]])
visualize_transformation(v, S0_50_5, 'S0.5,0.5')

S1m1 = np.array([[1, 0], [0, -1]])
visualize_transformation(v, S1m1, 'S1,-1 (Shear)')

Sm11 = np.array([[-1, 0], [0, 1]])
visualize_transformation(v, Sm11, 'S-1,1 (Reflection)')
