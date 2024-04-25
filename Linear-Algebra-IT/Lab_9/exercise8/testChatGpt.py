import numpy as np

def rotation_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

theta_a = np.pi
R_pi = rotation_matrix(theta_a)

theta_b = np.pi / 3
R_pi_over_3 = rotation_matrix(theta_b)

print("Rotation matrix Rπ:")
print(R_pi)
print("\nRotation matrix Rπ/3:")
print(R_pi_over_3)
