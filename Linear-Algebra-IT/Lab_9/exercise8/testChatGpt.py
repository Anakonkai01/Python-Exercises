import numpy as np

# Define the rotation matrices
def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

# (a) Describe geometrically the action of Rθ with θ = π
theta_a = np.pi
R_pi = rotation_matrix(theta_a)

# (b) Describe geometrically the action of Rθ with θ = π/3
theta_b = np.pi / 3
R_pi_over_3 = rotation_matrix(theta_b)

# Print the rotation matrices
print("Rotation matrix Rπ:")
print(R_pi)
print("\nRotation matrix Rπ/3:")
print(R_pi_over_3)
