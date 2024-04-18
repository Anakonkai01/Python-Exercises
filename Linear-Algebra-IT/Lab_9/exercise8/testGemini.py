import numpy as np
import matplotlib.pyplot as plt

def rotate_vector(v, theta):
  """
  This function rotates a 2D vector v by an angle theta using the rotation matrix Rθ.

  Args:
      v: A NumPy array representing the 2D vector (x, y).
      theta: The rotation angle in radians.

  Returns:
      A NumPy array representing the rotated vector (x', y').
  """
  R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
  return np.dot(R, v)

def visualize_rotation(v, theta, label):
  """
  This function visualizes the original and rotated vectors with arrows.

  Args:
      v: A NumPy array representing the original 2D vector (x, y).
      theta: The rotation angle in radians.
      label: A string label for the rotation.
  """
  origin = np.array([0, 0])  # Origin point
  rotated_v = rotate_vector(v, theta)

  # Plot arrows for original and rotated vectors
  plt.arrow(origin[0], origin[1], v[0], v[1], head_width=0.1, head_length=0.1, label='Original')
  plt.arrow(origin[0], origin[1], rotated_v[0], rotated_v[1], head_width=0.1, head_length=0.1, label=label, color='red')

  # Set axis limits slightly larger than vector coordinates
  plt.xlim(-max(abs(v[0]), abs(rotated_v[0])) * 1.1, max(abs(v[0]), abs(rotated_v[0])) * 1.1)
  plt.ylim(-max(abs(v[1]), abs(rotated_v[1])) * 1.1, max(abs(v[1]), abs(rotated_v[1])) * 1.1)

  # Label axes and add title
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.title(label + ' Rotation')
  plt.legend()
  plt.grid(True)
  plt.show()

# Define a sample vector
v = np.array([2, 1])  # You can change this vector

# Examples of rotations with visualizations
theta_pi = np.pi  # 180 degrees (reflection)
visualize_rotation(v, theta_pi, 'Rotation by 180° (Reflection)')

theta_pi_3 = np.pi / 3  # 60 degrees (counter-clockwise)
visualize_rotation(v, theta_pi_3, 'Rotation by 60° (Counter-Clockwise)')
